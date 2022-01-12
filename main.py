import tensorflow as tf
import tensorflow.keras as keras
import matplotlib 
import trimesh
import trimesh.exchange.xyz, trimesh.points
from discriminator import Discriminator
from generator import Generator
import numpy as np


# ====== GLOBAL HYPERPARAMS ===========
epochs = 1
batch_sz = 2
learning_rate = 0.0001
per_point_loss_weight = 0.1
num_points = 1024
latent_dim = 100
num_examples = 4
g_optimizer = keras.optimizers.Adam(learning_rate)
d_optimizer = keras.optimizers.Adam(learning_rate)

# ====== DATA PREPROCESSING ========
# read in meshes and convert to point clouds
data = []
for i in range(num_examples):
    path = "./blueno/blueno_" + str(i+1) + ".off"
    mesh = trimesh.load(path)
    # print(mesh.vertices.shape, mesh.faces.shape, mesh.triangles.shape, mesh.bounds)
    # cloud = trimesh.points.PointCloud(mesh.sample(num_points))
    # cloud.show()
    data.append(mesh.sample(num_points)) #[N,3]
# convert data to TF Dataset object and batch
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.batch(batch_sz, drop_remainder=True)

# read in FIXED sphere points
file = open("sphere_" + str(num_points) +"_points.xyz")
sphere = trimesh.exchange.xyz.load_xyz(file)['vertices'] #verts only

sphere = tf.reshape(sphere, [num_points, 3]) #[N,3]

# ====== SINGLE TRAINING STEP ===============
def train_batch(real_clouds):
    """
    trains D and G successively for one batch
    
    real_clouds: [B, N, 3]
    returns: d_loss (scalar), g_loss (scalar), generated_clouds [B,N,3]
    """
    # sample random latent vects from N(0,1)
    noise = tf.random.normal([batch_sz, latent_dim])
    spheres = tf.repeat(tf.expand_dims(sphere, axis=0), batch_sz, axis=0) #[B,N,3]
    # generate fake images
    fake_clouds = G(spheres, noise)

    # train D with real and fake clouds
    with tf.GradientTape() as tape:
        print("real:", tf.shape(real_clouds))
        real_shape_score, real_per_point_score = D(real_clouds)
        fake_shape_score, fake_per_point_score = D(fake_clouds)
        d_loss = D.loss(real_shape_score, real_per_point_score, fake_shape_score, fake_per_point_score)
    grads = tape.gradient(d_loss, D.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, D.trainable_variables))

    # train G with fake clouds
    with tf.GradientTape() as tape:
        fake_shape_score, fake_per_point_score = D(G(spheres, noise))
        g_loss = G.loss(fake_shape_score, fake_per_point_score)
    grads = tape.gradient(g_loss, G.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, G.trainable_variables))
    
    return d_loss, g_loss, fake_clouds

# ====== MAIN LOOP ==========
D = Discriminator(num_points, per_point_loss_weight)
G = Generator(num_points, latent_dim, per_point_loss_weight)
for epoch in range(epochs):
    print("Epoch: ", epoch)
    for batch_num, real_cloud_batch in enumerate(dataset):
        print("batch: ", batch_num)
        d_loss, g_loss, generated_clouds = train_batch(real_cloud_batch)
print("saving")
G.save("trained_generator1")

