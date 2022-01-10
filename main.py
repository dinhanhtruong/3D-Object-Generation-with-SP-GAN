import tensorflow as tf
import tensorflow.keras as keras
import matplotlib 
from discriminator import Discriminator
from generator import Generator

# ====== GLOBAL HYPERPARAMS ===========
epochs = 1
batch_sz = 64
learning_rate = 0.0001
num_points = 1024
latent_dim = 100
g_optimizer = keras.optimizers.Adam(learning_rate)
d_optimizer = keras.optimizers.Adam(learning_rate)

# ====== DATA PREPROCESSING ========
# read in meshes and convert to point clouds

# convert data to TF Dataset object

# read in sphere points

# ====== SINGLE TRAINING STEP ===============



# ====== MAIN LOOP ==========
D = Discriminator(__)
G = Generator(__)
for epoch in range(epochs):
    print("Epoch: ", epoch)


# save model
