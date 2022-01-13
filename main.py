import os
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot
import trimesh
import trimesh.exchange.xyz, trimesh.points
from discriminator import Discriminator
from generator import Generator
import numpy as np


# ====== GLOBAL HYPERPARAMS ===========
epochs = 10
batch_sz = 32
learning_rate_g = 0.0002
learning_rate_d = 0.0001
per_point_loss_weight = 0.1
num_points = 1024
latent_dim = 50
num_examples = 10000
d_optimizer = keras.optimizers.Adam(learning_rate_d, beta_1=0.5)
g_optimizer = keras.optimizers.Adam(learning_rate_g, beta_1=0.5)

# ====== DATA PREPROCESSING ========
# read in meshes and convert to point clouds
data = []
for i in range(num_examples):
    path = "./blueno/blueno_0.off" #"./blueno/blueno_" + str(i) + ".off"
    mesh = trimesh.load(path)
    # cloud = trimesh.points.PointCloud(mesh.sample(num_points))
    # cloud.show()
    # ==================================NORMALIZE =============================================================================? ###
    data.append(mesh.sample(num_points)) #[N,3]
# convert data to TF Dataset object and batch
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.batch(batch_sz, drop_remainder=True)

# read in FIXED sphere points
file = open("sphere_" + str(num_points) +"_points.xyz")
sphere = trimesh.exchange.xyz.load_xyz(file)['vertices'] #verts only
# cloud = trimesh.points.PointCloud(sphere)
# cloud.show()
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
        real_shape_score, real_per_point_score = D(real_clouds)
        fake_shape_score, fake_per_point_score = D(fake_clouds)
        print("real score: ", real_shape_score)
        print("fake score: ", fake_shape_score)
        d_loss = D.loss(real_shape_score, real_per_point_score, fake_shape_score, fake_per_point_score)
    grads = tape.gradient(d_loss, D.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, D.trainable_variables))

    noise = tf.random.normal([batch_sz, latent_dim])
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
# checkpoints to occasionally save model (generator only)
checkpoint = tf.train.Checkpoint(G=G) 
checkpoint_dir_prefix = "training_checkpoints/checkpoint"
G_losses = []
D_losses = []
for epoch in range(epochs):
    print("================ Epoch: ", epoch+1)
    for batch_num, real_cloud_batch in enumerate(dataset):
        print("--------batch: ", batch_num)
        d_loss, g_loss, generated_clouds = train_batch(real_cloud_batch)
        print("d_loss: ", d_loss)
        print("g_loss: ", g_loss)
        G_losses.append(g_loss)
        D_losses.append(d_loss)

        # plot losses per epoch 
        pyplot.cla()
        pyplot.plot(G_losses, label='generator')
        pyplot.plot(D_losses, label='discriminator')
        pyplot.xlabel("batch")
        pyplot.ylabel("loss")
        pyplot.legend()
        pyplot.title("G vs. D losses per epoch")
        pyplot.pause(0.05) #update plot
        if batch_num % 50 == 0:
            generated_clouds = tf.make_tensor_proto(generated_clouds)
            generated_clouds = trimesh.points.PointCloud(tf.make_ndarray(generated_clouds)[0])
            generated_clouds.show()
    
            
    print("saving")
    path = checkpoint.save(checkpoint_dir_prefix)
    print("path:", path)
G.summary()
pyplot.show()


