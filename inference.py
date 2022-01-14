import os
from tabnanny import check
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib 
import trimesh
import trimesh.exchange.xyz, trimesh.points
from discriminator import Discriminator
from generator import Generator
import numpy as np


# ====== GLOBAL HYPERPARAMS ===========
batch_sz = 32 # MUST MATCH TRAINING
per_point_loss_weight = 0.1 # MUST MATCH TRAINING
num_points = 1024 # MUST MATCH TRAINING
latent_dim = 70 #85 # MUST MATCH TRAINING
num_examples = 6

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
print("loading checkpoint at " + tf.train.latest_checkpoint(checkpoint_path))
status = checkpoint.restore("training_checkpoints/checkpoint-13")  # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
status.assert_consumed() # assert that all params loaded

# path = "./blueno/blueno_0.off" #"./blueno/blueno_" + str(i) + ".off"
# mesh = trimesh.load(path)
# cloud = trimesh.points.PointCloud(mesh.sample(num_points))
# cloud.show()


# infer
noise = tf.random.normal([batch_sz, latent_dim])
spheres = tf.repeat(tf.expand_dims(sphere, axis=0), batch_sz, axis=0) #[B,N,3]
fake_clouds = G(spheres, noise)
# visualize each cloud
for i, cloud in enumerate(tf.unstack(fake_clouds)):
    cloud = tf.make_tensor_proto(cloud)
    cloud = trimesh.points.PointCloud(tf.make_ndarray(cloud))
    cloud.show()
    if i == batch_sz-1:
        break