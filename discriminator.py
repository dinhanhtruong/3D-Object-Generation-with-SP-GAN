import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential

class Discriminator(keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, cloud):
        """
        Scores the input cloud with a scalar in [0,1] where 1 is 100% confidence that the input comes from the real data distribution.
        Score is twofold: consists of holistic shape score AND per-point score
        cloud: point cloud input [batch_sz, num_points, 3]
        returns: tuple of (per-shape score, per-point score) with size 1x1 and Nx1 respectively
        """
    