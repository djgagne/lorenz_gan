from keras.layers import concatenate, RepeatVector, Input, Permute
from keras.layers import Conv1D, UpSampling1D, Dense, Activation, Dropout, Reshape
from keras.optimizers import Adam
from keras.models import Model
import numpy as np


def generator_conv(num_cond_inputs, num_random_inputs, num_outputs, 
                   activation, min_conv_filters=8, min_data_width=4, filter_width=3):
    num_layers = int(np.log2(num_outputs) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers - 1))
    curr_conv_filters = max_conv_filters
    gen_cond_input = Input(shape=(num_cond_inputs, ))
    gen_rand_input = Input(shape=(num_random_inputs, ))
    gen_model = concatenate([gen_cond_input, gen_rand_input])
    gen_model = Dense(min_data_width * max_conv_filters)(gen_model)
    gen_model = Reshape((min_data_width, max_conv_filters))(gen_model)
    for i in range(0, num_layers):
        curr_conv_filters //= 2
        gen_model = Conv1D(curr_conv_filters, filter_width, padding="same")(gen_model)
        gen_model = Activation(activation)(gen_model)
        gen_model = UpSampling1D(size=2)(gen_model)
    gen_model = Conv1D(1, filter_width, padding="same")(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def discriminator_conv(num_cond_inputs, num_sample_inputs, activation, min_conv_filters=8, min_data_width=4,
		       filter_width=3):		
    disc_cond_input = Input(shape=(num_cond_inputs, ))
    disc_cond_input_repeat = RepeatVector(num_sample_inputs)(disc_cond_input)
    disc_cond_input_repeat = Permute((2, 1))(disc_cond_input_repeat)
    disc_sample_input = Input(shape=(num_sample_inputs, 1))
    disc_model = concatenate([disc_sample_input, disc_cond_input]) 
    num_layers = int(np.log2(num_sample_inputs) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers - 1))
    curr_conv_filters = max_conv_filters
    for i in range(num_layers):
        disc_model = Conv1D(curr_conv_filters, filter_width, strides=2, padding="same")(disc_model)
        disc_model = Activation(activation)(disc_model)
        curr_conv_filters *= 2
    disc_model = Flatten()(disc_model)
    disc_model = Dense(1)(disc_model)
    disc_model = Activation("sigmoid")(disc_model)
    discriminator = Model([disc_cond_input, disc_sample_input], disc_model)
    return discriminator


def generator_dense(num_cond_inputs, num_random_inputs, num_hidden, num_outputs, activation): 
    gen_cond_input = Input(shape=(num_cond_inputs, ))
    gen_rand_input = Input(shape=(num_random_inputs, ))
    gen_model = concatenate([gen_cond_input, gen_rand_input])
    gen_model = Dense(num_hidden)(gen_model)
    gen_model = Activation(activation)(gen_model)
    gen_model = Dense(num_outputs)(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def discriminator_dense(num_cond_inputs, num_sample_inputs, num_hidden, activation):
    disc_cond_input = Input(shape=(num_cond_inputs, ))
    disc_sample_input = Input(shape=(num_sample_inputs, ))
    disc_model = concatenate([disc_cond_input, disc_sample_input])
    disc_model = Dense(num_hidden)(disc_model)
    disc_model = Activation(activation)(disc_model)
    disc_model = Dense(1)(disc_model)
    disc_model = Actvation("sigmoid")(disc_model)
    discriminator = Model([disc_cond_input, disc_sample_input], disc_model)
    return discriminator

def stack_gen_disc(generator, discriminator):
    """
    Combines generator and discrminator layers together while freezing the weights of the discriminator layers

    Args:
        generator:
        discriminator:

    Returns:
        Generator layers attached to discriminator layers.
    """
    discriminator.trainable = False
    model = discriminator(generator.output)
    model_obj = Model(generator.input, model)
    return model_obj


def initialize_gan(generator, discriminator, optimizer):
    generator.compile(optimizer=optimizer, loss="mse")
    discriminator.compile(optimizer=optimizer, loss="binary_crossentropy")
    gen_disc = stack_gen_disc(generator, discriminator)
    gen_disc.compile(optimizer=optimizer, loss="binary_crossentropy")
    print(generator.summary())
    print(discriminator.summary())
    print(gen_disc.summary())
    return gen_disc

def train_gan(data, generator, discriminator, batch_size, num_epochs):
    return
