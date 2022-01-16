import tensorflow as tf
import trimesh
import trimesh.exchange.xyz, trimesh.points
from discriminator import Discriminator
from generator import Generator
import numpy as np

num_examples = 5
interpolation_steps = 4


# ====== GLOBAL HYPERPARAMS - DO NOT CHANGE ===========
batch_sz = 16 # MUST MATCH TRAINING
per_point_loss_weight = 0.4 # MUST MATCH TRAINING
num_points = 2048 # MUST MATCH TRAINING
latent_dim = 100 #85 # MUST MATCH TRAINING


# read in FIXED sphere points
file = open("sphere_" + str(num_points) +"_points.xyz")
sphere = trimesh.exchange.xyz.load_xyz(file)['vertices'] #verts only
sphere = tf.reshape(sphere, [num_points, 3]) #[N,3]

# load in model
D = Discriminator(num_points, per_point_loss_weight)
G = Generator(num_points, latent_dim, per_point_loss_weight)

# run model once to configure for restoring
noise = tf.random.normal([batch_sz, latent_dim], 0, 1)
spheres = tf.repeat(tf.expand_dims(sphere, axis=0), batch_sz, axis=0) #[B,N,3]
fake_clouds = G(spheres, noise)

checkpoint = tf.train.Checkpoint(G=G)
checkpoint_path = "trained_generator"
print("loading checkpoint at " + tf.train.latest_checkpoint(checkpoint_path))
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))  #   
status.assert_consumed() # assert that all params loaded


# returns np array of blueno coords
def getBlueno(noise):
    blueno = G(spheres, noise)
    blueno = tf.make_tensor_proto(tf.squeeze(blueno))
    blueno = tf.make_ndarray(blueno)[0]
    return blueno

# generate 2 distinct bluenos
spheres = tf.repeat(tf.expand_dims(sphere, axis=0), batch_sz, axis=0) #[B,N,3]
noise1 = tf.random.normal([batch_sz, latent_dim], 0, 0.5)
noise2 = tf.random.normal([batch_sz, latent_dim], 0, 3)

blueno1 = getBlueno(noise1)
blueno2 = getBlueno(noise2)

print("interpolating:")
# interpolate
for i in range(interpolation_steps):
    a = float(i)/interpolation_steps
    intermediate_blueno = (1-a)*blueno1 + a*blueno2
    intermediate_blueno = trimesh.points.PointCloud(intermediate_blueno)
    intermediate_blueno.show()

# visualize each cloud
print("showing random bluenos:")
fake_clouds = G(spheres, noise)
for i, cloud in enumerate(tf.unstack(fake_clouds)):
    cloud = tf.make_tensor_proto(cloud)
    cloud = trimesh.points.PointCloud(tf.make_ndarray(cloud))
    cloud.show()
    if i == batch_sz-1 or i == num_examples:
        break

