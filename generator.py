import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, LeakyReLU, Softmax, GlobalMaxPool1D, Dense, Reshape, Embedding, BatchNormalization

class Generator(keras.Model):
    def __init__(self, num_points, latent_dim, per_point_loss_weight):
        super().__init__()
        self.feature_emb_sz = 128
        self.style_emb1_sz = 64
        self.style_emb2_sz = 128
        self.leaky_grad = 0.01
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.per_point_loss_weight = per_point_loss_weight

        # [B, N, (3+latent_dim)] -> [B,N, feature_emb_sz]
        self.feature_emb = Sequential([
            Conv1D(self.feature_emb_sz, kernel_size=1, input_shape=(num_points, 3+latent_dim)),
            LeakyReLU(self.leaky_grad),
            Conv1D(self.feature_emb_sz, kernel_size=1),
            LeakyReLU(self.leaky_grad),
        ])

        # [B,N, feature_emb_sz] ->  [B,N, 2*style_emb1/2_sz]
        self.style_emb1 = Conv1D(2*self.style_emb1_sz, kernel_size=1, input_shape=(num_points, self.style_emb1_sz))
        self.style_emb2 = Conv1D(2*self.style_emb1_sz, kernel_size=1, input_shape=(num_points, self.style_emb2_sz))

        # [B, N, dim_in] -> [B, N, dim_out]
        self.graph_attn1 = GraphAttention(dim_in=3, dim_out=self.style_emb1_sz, k=20, n=num_points)
        self.graph_attn2 = GraphAttention(self.style_emb1_sz, self.style_emb2_sz, 20, num_points)

        self.adaptive_instance_norm1 = AdaptiveInstanceNorm()
        self.adaptive_instance_norm2 = AdaptiveInstanceNorm()

        # [B, N, 128] -> [B,1,128]
        self.max_pool = GlobalMaxPool1D()

        # [B, 128] -> [B, 512]
        self.global_feature_MLP = Sequential([
            Dense(self.feature_emb_sz),
            BatchNormalization(),
            LeakyReLU(self.leaky_grad),
            Dense(512),
            BatchNormalization(),
            LeakyReLU(self.leaky_grad),
        ])

        # [B, N, (512 + self.style_emb2_sz)] -> [B, N, 3]
        self.MLP_out = Sequential([
            Conv1D(256, kernel_size=1, input_shape=(num_points, 3+self.style_emb1_sz)),
            LeakyReLU(self.leaky_grad),
            Conv1D(64, kernel_size=1),
            LeakyReLU(self.leaky_grad),
            Conv1D(3, kernel_size=1, activation='tanh'),
        ])
    def call(self, sphere, latent_vec):
        """
        sphere: [B, N, 3]
        latent_vec: [B, latent_dim]

        Returns: generated point cloud [B, N,3]
        """
        # 1) upper branch: get local style embedding from prior latent matrix
        latent_vecs = tf.expand_dims(latent_vec, 1) # [B, 1, latent_dim]
        latent_vecs = tf.repeat(latent_vecs, self.num_points, axis=1) # [B,N, latent_dim]
        latent_matrix = tf.concat([sphere, latent_vecs], axis=-1) # [B,N, 3+latent_dim]
        feature_emb = self.feature_emb(latent_matrix) # [B,N, feature_emb_sz]
        local_style1 = self.style_emb1(feature_emb) # [B,N, 2*style_emb1_sz]
        # 2) lower branch: apply graph attention module to get feature map
        feature_map = self.graph_attn1(sphere) # [B, N, 64]
        # 3) get embedded feature map: fuse local style with global feature map
        normalized_feature_map = self.adaptive_instance_norm1(feature_map, local_style1) # [B, N, 64]
        
        # repeat 1-3
        local_style2 = self.style_emb2(feature_emb) # [B,N, 2*style_emb2_sz]
        feature_map2 = self.graph_attn2(normalized_feature_map) # [B, N, 128]
        normalized_feature_map2 = self.adaptive_instance_norm2(feature_map2, local_style2) # [B, N, 128]
        
        # reconstruct point cloud from new embedded feature map
        pooled_features = self.max_pool(normalized_feature_map2) # [B, 1, 128]

        global_features = self.global_feature_MLP(tf.squeeze(pooled_features)) # [B, 512]
        global_features = tf.expand_dims(global_features, 1) # [B, 1, 512]
        duplicated_features = tf.repeat(global_features, self.num_points, axis=1) # [B, N, 512]
        concat = tf.concat([normalized_feature_map2, duplicated_features], axis=-1) # [B, N, (512+128)]
        return self.MLP_out(concat) # [B, N,3]
    def loss(self, fake_shape_scores, fake_per_point_scores):
        shape_loss = 0.5 * (fake_shape_scores-1)**2 # [B, 1]
        
        point_loss_inner_sum = (fake_per_point_scores-1)**2 # [B, N, 1]
        point_loss = 1.0/(2*self.num_points) * tf.reduce_sum(point_loss_inner_sum, axis=1)  # [B, 1]

        return tf.reduce_sum(shape_loss +  self.per_point_loss_weight * point_loss)
# helper classes
class GraphAttention(keras.Model):
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
        self.MLPs_upper = Sequential([
            Conv2D(dim_out, kernel_size=1, strides=1), # (B, N, k, dim_out)
            BatchNormalization(axis=-1), # channel last 
            LeakyReLU(0.01)
        ])
        # same as above with softmax
        self.MLPs_lower = Sequential([
            Conv2D(dim_out, kernel_size=1, strides=1), # (B, N, k, dim_out)
            BatchNormalization(axis=-1), # channel last 
            LeakyReLU(0.01),
            Softmax(axis=2)  #softmax along k axis
        ])

        # [B,N,k,C] -> [B,N, dim_out]
        self.conv_out = Sequential([
            Conv2D(dim_out, kernel_size=[1,k], strides=1, input_shape=(n, k)), # (B, N, dim_out)
        ])


    def call(self, x):
        """
        x: (point cloud) input [batch, N, C] where C is the dimension of the points in the cloud (C=3 initially)
        returns: point-wise feature map [B, N, dim_out]
        """
        # KNN grouping (lower branch)
        dist_adj_matrix = self.pairwise_distance(x) # builds adj matrix [B, N, N] as indices
        assert dist_adj_matrix.shape == (64, 1024, 1024)

        nn_idx = self.knn(dist_adj_matrix) # [B, N, k]
        assert nn_idx.shape == (64, 1024, self.k)
        
        upper_branch, lower_branch = self.get_edge_feature(tf.expand_dims(x, axis=2), nn_idx) # [B, N, k, 2C], [B, N, k, C]
        print("upper:", upper_branch.shape)
        print("lower:", lower_branch.shape)
        assert upper_branch.shape == (64, 1024, self.k, 2*x.shape[-1])
        assert lower_branch.shape == (64, 1024, self.k, x.shape[-1])

        # apply MLPs (EdgeConv) to both branches
        feature_map = self.MLPs_upper(upper_branch) # [B, N, k, dim_out]
        feature_weights = self.MLPs_lower(lower_branch) # [B, N, k, dim_out]

        # collapse upper/lower branches
        weighted_feature_map = feature_map * feature_weights # [B, N, k, dim_out]

        return self.conv_out(weighted_feature_map) # [B, N, dim_out]

    
    # ===== EdgeConv module from DGCNN ==============
    def get_edge_feature(self, point_cloud, nn_idx, k=20):
        """Construct edge feature for each point
        Args:
            point_cloud: (batch_size, num_points, 1, num_dims)
            nn_idx: (batch_size, num_points, k)
            k: int
        Returns:
            edge features: (batch_size, num_points, k, 2*num_dims)
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
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_) #KNN grouping
        point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

        point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1]) #duplicate k times

        edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
        return edge_feature, point_cloud_neighbors

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
        point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)
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

class AdaptiveInstanceNorm(keras.Model):
    def __init__(self):
        super().__init__()
        # normalize per feature vector/instance, not across entire batch
        self.norm = BatchNormalization(axis=[0,1])


    def call(self, feature_map, styles, training=True):
        """
        feature_map: output of graph attention, [B, N, dim_graph_attn_out]
        styles: output of style embedding, [B,N, 2*style_emb_sz]
        returns: normalized feature map (same size)
        """
        # split styles into scale and bias scalars
        scale, bias = tf.split(styles, 2) # [B, N, style_emb_sz]
        return scale * self.norm(feature_map) + bias