from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras import metrics

batch_size = 1024
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
eplisilon_std = 1.0


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=eplisilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon

x = Input(shape=(original_dim,), name="input")
h = Dense(intermediate_dim, activation="relu", name="encoder")(x)
z_mean = Dense(latent_dim, name="mean")(h)
z_log_var = Dense(latent_dim, name="log-varianse")(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(x, [z_mean, z_log_var, z], name="z_log_var")
encoder.summary()

input_decorder = Input(shape=(latent_dim,), name="decoder_input")
decorder_h = Dense(intermediate_dim, activation="relu", name="decoder_h")(input_decorder)
x_decoeded = Dense(original_dim, activation="sigmoid", name="flat_decorded")(decorder_h)
decorder = Model(input_decorder, x_decoeded, name="decorder")
decorder.summary()

output_combined = decorder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()

def vae_loss(x: tf.Tensor, x_decoeded_mean: tf.Tensor, z_log_var=z_log_var, z_mean=z_mean, original_dim=original_dim):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoeded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") /  255.
x_test = x_test.astype("float32") /  255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size)