import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, LeakyReLU, GlobalMaxPool1D, Dense, Reshape, BatchNormalization

class Discriminator(keras.Model):
    def __init__(self, num_points, per_point_loss_weight):
        super().__init__()
        self.num_points = num_points
        self.leaky_grad = 0.01
        self.per_point_loss_weight = per_point_loss_weight
        # normalize per feature vector/instance, not across entire batch

        self.feature_extraction = Sequential([
            Conv1D(64, kernel_size=1, input_shape=(num_points, 3)),
            BatchNormalization(axis=-1), #channel last
            LeakyReLU(self.leaky_grad),
            Conv1D(128, kernel_size=1),
            BatchNormalization(axis=-1),
            LeakyReLU(self.leaky_grad),
            Conv1D(256, kernel_size=1),
            BatchNormalization(axis=-1),
            LeakyReLU(self.leaky_grad),
            Conv1D(512, kernel_size=1),
            BatchNormalization(axis=-1),
            LeakyReLU(self.leaky_grad),
        ])

        # [B, N, C] -> [B,1,C]
        self.max_pool = GlobalMaxPool1D()

        # [B, (N), C] -> [B, (N), 1]
        self.MLPs_per_shape = Sequential([
            Dense(512),
            LeakyReLU(self.leaky_grad),
            Dense(256),
            LeakyReLU(self.leaky_grad),
            Dense(64),
            LeakyReLU(self.leaky_grad),
            Dense(1) # sigmoid????
        ])
        self.MLPs_per_point = Sequential([
            Dense(512),
            LeakyReLU(self.leaky_grad),
            Dense(256),
            LeakyReLU(self.leaky_grad),
            Dense(64),
            LeakyReLU(self.leaky_grad),
            Dense(1) # sigmoid????
        ])

        



    def call(self, cloud):
        """
        Scores the input cloud with a scalar in [0,1] where 1 is 100% confidence that the input comes from the real data distribution.
        Score is twofold: consists of holistic shape score AND per-point score
        
        cloud: point cloud input [batch_sz, num_points, 3]
        returns: tuple of (batch_sz, 1) and (batch_sz, num_points) containing per-shape and per-point scores, respectively
        """
        features = self.feature_extraction(cloud)

        # split into two branches
        pooled = self.max_pool(features)
        per_shape_score = self.MLPs_per_shape(pooled) 
        per_shape_score = tf.squeeze(per_shape_score) # [B,1]

        per_point_score = self.MLPs_per_point(features) 
        per_point_score = tf.squeeze(per_point_score) # [B,N]

        return [per_shape_score, per_point_score]

    def loss(self, real_shape_scores, real_per_point_scores, fake_shape_scores, fake_per_point_scores):
        """
        real/fake_shape_scores: [B, 1]
        real/fake_per_point_scores: [B, N]
        returns: batch loss (scalar)
        """
        shape_loss = 0.5 * (fake_shape_scores**2 +(real_shape_scores-1)**2) # [B, 1]
        
        point_loss_inner_sum = (fake_per_point_scores**2 +(real_per_point_scores-1)**2) # [B, N, 1]
        point_loss = 1.0/(2*self.num_points) * tf.reduce_sum(point_loss_inner_sum, axis=1)  # [B, 1]

        return tf.reduce_sum(shape_loss +  self.per_point_loss_weight * point_loss)