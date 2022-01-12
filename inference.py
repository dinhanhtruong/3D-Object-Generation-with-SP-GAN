import os
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib 
import trimesh
import trimesh.exchange.xyz, trimesh.points
from discriminator import Discriminator
from generator import Generator
import numpy as np


# ====== GLOBAL HYPERPARAMS ===========
batch_sz = 2
per_point_loss_weight = 0.1
num_points = 1024 # MUST MATCH TRAINING
latent_dim = 100
num_examples = 4

# read in FIXED sphere points
file = open("sphere_" + str(num_points) +"_points.xyz")
sphere = trimesh.exchange.xyz.load_xyz(file)['vertices'] #verts only
sphere = tf.reshape(sphere, [num_points, 3]) #[N,3]

# load in model
D = Discriminator(num_points, per_point_loss_weight)
G = Generator(num_points, latent_dim, per_point_loss_weight)

# run model once to configure for restoring
noise = tf.random.normal([batch_sz, latent_dim])
spheres = tf.repeat(tf.expand_dims(sphere, axis=0), batch_sz, axis=0) #[B,N,3]
fake_clouds = G(spheres, noise)


checkpoint = tf.train.Checkpoint(G=G)
checkpoint_path = "training_checkpoints"
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
status.assert_consumed() # assert that all params loaded

# infer
noise = tf.random.normal([batch_sz, latent_dim])
spheres = tf.repeat(tf.expand_dims(sphere, axis=0), batch_sz, axis=0) #[B,N,3]
fake_clouds = G(spheres, noise)
# visualize each cloud
for cloud in tf.unstack(fake_clouds):
    cloud = proto_tensor = tf.make_tensor_proto(cloud)
    cloud = trimesh.points.PointCloud(tf.make_ndarray(cloud))
    cloud.show()