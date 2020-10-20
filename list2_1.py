from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np

batch_size = 1024
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epoch = 50
eplisilon_std = 1.0

x = Input(shape=(original_dim,), name="input")
h = Dense(intermediate_dim, activation="relu", name="encoder")(x)
z_mean = Dense(latent_dim, name="mean")(h)
z_log_var = Dense(latent_dim, name="log-varianse")(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(x, [z_mean, z_log_var, z], name="z_log_var")

def sampling(argd: tupple):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean)[0], latent_dim)

