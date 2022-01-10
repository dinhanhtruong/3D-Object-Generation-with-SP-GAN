import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, LeakyReLU, Dense, Reshape, Embedding, BatchNormalization

class Generator(keras.Model):
    def __init__(self, num_points, latent_dim):
        super().__init__()
        self.feature_emb_sz = 128
        self.style_emb1_sz = 64
        self.style_emb2_sz = 128

        # [B, N, (3+latent_dim)] -> [B,N, feature_emb_sz]
        self.feature_emb = Sequential([
            Conv1D(self.feature_emb_sz, kernel_size=1, input_shape=(num_points, 3+latent_dim)),
            LeakyReLU(0.01),
            Conv1D(self.feature_emb_sz, kernel_size=1),
            LeakyReLU(0.01),
        ])

        # [B,N, feature_emb_sz] ->  [B,N, 2*]
        self.style_emb1 = Sequential([
            Conv1D(self.feature_emb_sz, kernel_size=1, input_shape=(num_points, 3+latent_dim)),
            LeakyReLU(0.01),
            Conv1D(self.feature_emb_sz, kernel_size=1),
            LeakyReLU(0.01),
        ])

        self.graph_attn1 = Graph_Attention(3, 64, 20, 1024)
        self.graph_attn2 = Graph_Attention(64, 128, 20, 1024)

        self.adaptive_instance_norm = _

    def call(self, sphere, latent_vec):
        """
        sphere: [N, 3]
        latent_vec: [latent_dim,]

        Returns: generated point cloud [N,3]
        """
        # 1) get local style embedding from prior latent matrix

        # 2) lower branch: apply graph attention module to get feature map

        # 3) get embedded feature map: fuse local style with global feature map

        # repeat 1-3

        # reconstruct point cloud from new embedded feature map

# helper class
class Graph_Attention(keras.Model):
    def __init__(self, dim_in, dim_out, k, n):
        super().__init__()

        self.k = k #20
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # upper branch
        # self.conv_features = Sequential([
        #     Conv2D(dim_out, kernel_size=1, strides=1, input_shape=(n, k)), # (B, dim_out, N, k)
        #     BatchNormalization(axis=1),
        #     LeakyReLU(self.leaky_grad)
        # ])

        # [B,N,k,_] -> [B,N,k,dim_out]
        self.MLPs = Sequential([
            Conv2D(dim_out, kernel_size=1, strides=1), # (B, N, k, dim_out)
            BatchNormalization(axis=-1), # channel last 
            LeakyReLU(0.01)
        ])

        # [B,N,k,C] -> [B,N, dim_out]
        self.conv_out = Sequential([
            Conv2D(dim_out, kernel_size=[1,k], strides=1, input_shape=(n, k)), # (B, N, dim_out)
        ])


    def call(self, x):
        """
        x: point cloud input [batch, N, C] where C is the dimension of the points in the cloud (C=3)
        returns: point-wise feature map [B, N, 2*dim_out]
        """
        
        # duplicate K times (upper branch)
        upper_branch = tf.expand_dims(x, axis=2) # [B, N, 1, C]
        upper_branch = tf.repeat(x, self.k, axis=2) # [B, N, k, C]

        # KNN grouping (lower branch)
        lower_branch = self.pairwise_distance(x) # builds adj matrix [B, N, N] as indices
        lower_branch = self.knn(lower_branch) # [B, N, k]
        lower_branch = self.get_edge_feature # [B, N, k, C]
        lower_branch -= upper_branch

        # concatenate upper branch (duplicated cloud) with the adjusted KNN grouping
        upper_branch = tf.concat([upper_branch, lower_branch], axis=-1 ) # [B, N, k, 2C]

        # apply MLPs (EdgeConv) to both branches
        upper_branch = self.MLPs(upper_branch) # [B, N, k, dim_out]

        lower_branch = self.MLPs(lower_branch) # [B, N, k, dim_out]
        feature_weights = tf.nn.softmax(lower_branch, axis=2) #softmax along k axis

        # collapse upper/lower branches
        weighted_feature_map = upper_branch * feature_weights # [B, N, k, dim_out]

        return self.conv_out(weighted_feature_map) # [B, N, dim_out]

    
    # ===== EdgeConv module from DGCNN ==============
    def get_edge_feature(self, point_cloud, nn_idx, k=20):
        """Construct edge feature for each point
        Args:
            point_cloud: (batch_size, num_points, 1, num_dims)
            nn_idx: (batch_size, num_points, k)
            k: int
        Returns:
            edge features: (batch_size, num_points, k, num_dims)
        """
        og_batch_size = point_cloud.shape[0]
        point_cloud = tf.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = tf.expand_dims(point_cloud, 0)

        point_cloud_central = point_cloud

        point_cloud_shape = point_cloud.shape
        batch_size = point_cloud_shape[0]
        num_points = point_cloud_shape[1]
        num_dims = point_cloud_shape[2]

        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

        point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
        point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

        point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

        edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
        return edge_feature

    def pairwise_distance(self, point_cloud):
        """Compute pairwise distance of a point cloud.
        Args:
            point_cloud: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        og_batch_size = point_cloud.shape[0]
        point_cloud = tf.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = tf.expand_dims(point_cloud, 0)
            
        point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
        point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2*point_cloud_inner
        point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
        point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
        return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

    def knn(self, adj_matrix, k=20):
        """Get KNN based on the pairwise distance.
        Args:
            pairwise distance: (batch_size, num_points, num_points)
            k: int
        Returns:
            nearest neighbors: (batch_size, num_points, k)
        """
        neg_adj = -adj_matrix
        _, nn_idx = tf.nn.top_k(neg_adj, k=k)
        return nn_idx