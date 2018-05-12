from keras.layers import concatenate, RepeatVector, Input, Flatten, BatchNormalization, Dropout, AlphaDropout
from keras.layers import Conv1D, Activation, Reshape, LeakyReLU, Layer, MaxPool1D, AveragePooling1D
from keras.layers import GaussianNoise
from keras.initializers import RandomUniform
import keras.backend as K
from keras.optimizers import Adam
import xarray as xr
from scipy.stats import expon
from keras.models import Model
import numpy as np
import pandas as pd
from os.path import join
from keras.regularizers import l2
from keras.engine import InputSpec
from keras.layers import Dense, Wrapper


class Interpolate1D(Layer):
    def __init__(self, **kwargs):
        self.size = int(2)
        super(Interpolate1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        size = self.size * input_shape[1] if input_shape[1] is not None else None
        return input_shape[0], size, input_shape[2]

    def call(self, inputs, **kwargs):
        in_shape = K.int_shape(inputs)
        combined_values = []
        total_size = in_shape[1] * self.size
        j = 0
        for i in range(total_size):
            if i == total_size - 1:
                combined_values.append(2 * inputs[:, j-1: j, :] - combined_values[-1])
            elif i % 2 == 0:
                combined_values.append(inputs[:, j: j + 1, :])
                j += 1
            else:
                combined_values.append(0.5 * inputs[:, j:j + 1, :] + 0.5 * inputs[:, j - 1: j, :])
        output = K.concatenate(combined_values, axis=1)
        return output


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        input_dim = np.prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


def predict_stochastic(neural_net):
    """
    Have the neural network make predictions with the Dropout layers on, resulting in stochastic behavior from the
    neural net itself.

    Args:
        neural_net:
        data:

    Returns:

    """
    input = neural_net.input
    output = neural_net.output
    pred_func = K.function(input + [K.learning_phase()], [output])
    return pred_func


def generator_conv_concrete(num_cond_inputs=3, num_random_inputs=10, num_outputs=32,
                   activation="selu", min_conv_filters=8, min_data_width=4, filter_width=3,
                   dropout_alpha=0.2, data_size=1.0e5, normalize=True):
    """
    Convolutional conditional generator neural network. The conditional generator takes a combined vector of
    normalized conditional and random values and outputs normalized synthetic examples of the training data.
    This function creates the network architecture. The network design assumes that the number of convolution
    filters halves with each convolutional layer.

    Args:
        num_cond_inputs (int): Size of the conditional input vector.
        num_random_inputs (int): Size of the random input vector.
        num_outputs (int): Size of the output vector. Recommend using a power of 2 for easy scaling.
        activation (str): Type of activation function for the convolutional layers. Recommend selu, elu, or relu.
        min_conv_filters (int): Number of convolutional filters at the second to last convolutional layer
        min_data_width (int): Width of the first convolutional layer after the dense layer. Should be a power
            of 2.
        filter_width (int): Width of the convolutional filters
    Returns:
        generator: Keras Model object of the generator network
    """
    l = 1e-4
    wd = l ** 2. / data_size
    dd = 2. / data_size
    num_layers = int(np.log2(num_outputs) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers))
    curr_conv_filters = max_conv_filters
    gen_cond_input = Input(shape=(num_cond_inputs, ))
    gen_cond_repeat = RepeatVector(min_data_width)(gen_cond_input)
    gen_rand_input = Input(shape=(num_random_inputs, ))
    gen_model = Reshape((num_random_inputs, 1))(gen_rand_input)
    gen_model = ConcreteDropout(Conv1D(max_conv_filters, filter_width, padding="same",
                                       kernel_regularizer=l2()),
                                weight_regularizer=wd, dropout_regularizer=dd)(gen_model)
    if activation == "leaky":
        gen_model = LeakyReLU(0.2)(gen_model)
    else:
        gen_model = Activation(activation)(gen_model)
    gen_model = concatenate([gen_model, gen_cond_repeat])
    data_width = min_data_width
    for i in range(0, num_layers):
        curr_conv_filters //= 2
        if data_width < filter_width:
            f_width = 3
        else:
            f_width = filter_width
        gen_model = ConcreteDropout(Conv1D(curr_conv_filters, f_width,
                                           padding="same", kernel_regularizer=l2()),
                                    weight_regularizer=wd, dropout_regularizer=dd)(gen_model)
        if activation == "leaky":
            gen_model = LeakyReLU(0.2)(gen_model)
        else:
            gen_model = Activation(activation)(gen_model)
        gen_model = Interpolate1D()(gen_model)
        data_width *= 2
    gen_model = Conv1D(1, filter_width, padding="same", kernel_regularizer=l2())(gen_model)
    if normalize:
        gen_model = BatchNormalization()(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def generator_conv(num_cond_inputs=3, num_random_inputs=10, num_outputs=32,
                   activation="selu", min_conv_filters=8, min_data_width=4, filter_width=3,
                   dropout_alpha=0.2, normalize=True):
    """
    Convolutional conditional generator neural network. The conditional generator takes a combined vector of
    normalized conditional and random values and outputs normalized synthetic examples of the training data.
    This function creates the network architecture. The network design assumes that the number of convolution
    filters halves with each convolutional layer.

    Args:
        num_cond_inputs (int): Size of the conditional input vector.
        num_random_inputs (int): Size of the random input vector.
        num_outputs (int): Size of the output vector. Recommend using a power of 2 for easy scaling.
        activation (str): Type of activation function for the convolutional layers. Recommend selu, elu, or relu.
        min_conv_filters (int): Number of convolutional filters at the second to last convolutional layer
        min_data_width (int): Width of the first convolutional layer after the dense layer. Should be a power
            of 2.
        filter_width (int): Width of the convolutional filters
        dropout_alpha (float): Proportion of neurons randomly set to 0.
        normalize (bool): Whether to include a batch normalization layer at the end of the model.
    Returns:
        generator: Keras Model object of the generator network
    """
    num_layers = int(np.log2(num_outputs) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers))
    curr_conv_filters = max_conv_filters
    gen_cond_input = Input(shape=(num_cond_inputs, ))
    gen_cond_repeat = RepeatVector(min_data_width)(gen_cond_input)
    gen_rand_input = Input(shape=(num_random_inputs, ))
    if num_random_inputs == min_data_width:
        gen_model = Reshape((num_random_inputs, 1))(gen_rand_input)
    else:
        num_random_filters = num_random_inputs // min_data_width
        gen_model = Reshape((min_data_width, num_random_filters))(gen_rand_input)
    gen_model = Conv1D(max_conv_filters, filter_width, padding="same", kernel_regularizer=l2())(gen_model)
    if activation == "leaky":
        gen_model = LeakyReLU(0.2)(gen_model)
    else:
        gen_model = Activation(activation)(gen_model)
    gen_model = Dropout(dropout_alpha)(gen_model)
    gen_model = concatenate([gen_model, gen_cond_repeat])
    data_width = min_data_width
    for i in range(0, num_layers):
        curr_conv_filters //= 2
        if data_width < filter_width:
            f_width = 3
        else:
            f_width = filter_width
        gen_model = Conv1D(curr_conv_filters, f_width, padding="same", kernel_regularizer=l2())(gen_model)
        if activation == "leaky":
            gen_model = LeakyReLU(0.2)(gen_model)
        else:
            gen_model = Activation(activation)(gen_model)
        if activation == "selu":
            gen_model = Dropout(dropout_alpha)(gen_model)
        else:
            gen_model = Dropout(dropout_alpha)(gen_model)
        gen_model = Interpolate1D()(gen_model)
        data_width *= 2
    gen_model = Conv1D(1, filter_width, padding="same", kernel_regularizer=l2())(gen_model)
    if normalize:
        gen_model = BatchNormalization()(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def discriminator_conv_concrete(num_cond_inputs=3, num_sample_inputs=32, activation="selu",
                       min_conv_filters=8, min_data_width=4,
                       filter_width=3, dropout_alpha=0.5, data_size=1.0e5):
    """
    Convolutional conditional discriminator neural network architecture. The conditional discriminator takes the
    conditional vector and a real or synthetic sample and predicts the probability that the sample is real or not.

    Args:
        num_cond_inputs (int): Size of the conditional input vector.
        num_sample_inputs (int): Size of the sample input vector
        activation (str): Activation Function for convolutional layers. Recommend selu, elu, or relu.
        min_conv_filters (int): Number of convolution filters in the first convolution layer. Doubles with
            each subsequent layer
        min_data_width (int): Width of the convolutional data after the last convolutional layer. Assumes halving
            from the initial vector
        filter_width (int): Width of the convolutional feature maps

    Returns:
        Discrminator Keras Model object
    """
    l = 1e-4
    wd = l ** 2. / data_size
    dd = 2. / data_size
    disc_cond_input = Input(shape=(num_cond_inputs, ))
    disc_cond_input_repeat = RepeatVector(num_sample_inputs)(disc_cond_input)
    disc_sample_input = Input(shape=(num_sample_inputs, 1))
    disc_model = concatenate([disc_sample_input, disc_cond_input_repeat])
    num_layers = int(np.log2(num_sample_inputs) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    for i in range(num_layers):
        disc_model = ConcreteDropout(Conv1D(curr_conv_filters, filter_width,
                                            strides=1, padding="same", kernel_regularizer=l2()),
                                     weight_regularizer=wd, dropout_regularizer=dd)(disc_model)
        if activation == "leaky":
            disc_model = LeakyReLU(0.2)(disc_model)
        else:
            disc_model = Activation(activation)(disc_model)
        disc_model = BatchNormalization()(disc_model)
        disc_model = AveragePooling1D()(disc_model)
        curr_conv_filters *= 2
    disc_model = ConcreteDropout(Conv1D(curr_conv_filters, filter_width,
                                        padding="same", kernel_regularizer=l2()),
                                 weight_regularizer=wd, dropout_regularizer=dd)(disc_model)
    if activation == "leaky":
        disc_model = LeakyReLU(0.2)(disc_model)
    else:
        disc_model = Activation(activation)(disc_model)
    disc_model = Flatten()(disc_model)
    disc_model = Dense(1, kernel_regularizer=l2())(disc_model)
    disc_model = Activation("sigmoid")(disc_model)
    discriminator = Model([disc_cond_input, disc_sample_input], disc_model)
    return discriminator


def discriminator_conv(num_cond_inputs=3, num_sample_inputs=32, activation="selu",
                       min_conv_filters=8, min_data_width=4,
                       filter_width=3, dropout_alpha=0.5):
    """
    Convolutional conditional discriminator neural network architecture. The conditional discriminator takes the
    conditional vector and a real or synthetic sample and predicts the probability that the sample is real or not.

    Args:
        num_cond_inputs (int): Size of the conditional input vector.
        num_sample_inputs (int): Size of the sample input vector
        activation (str): Activation Function for convolutional layers. Recommend selu, elu, or relu.
        min_conv_filters (int): Number of convolution filters in the first convolution layer. Doubles with
            each subsequent layer
        min_data_width (int): Width of the convolutional data after the last convolutional layer. Assumes halving
            from the initial vector
        filter_width (int): Width of the convolutional feature maps

    Returns:
        Discrminator Keras Model object
    """
    disc_cond_input = Input(shape=(num_cond_inputs, ))
    disc_cond_input_repeat = RepeatVector(num_sample_inputs)(disc_cond_input)
    disc_sample_input = Input(shape=(num_sample_inputs, 1))
    disc_model = concatenate([disc_sample_input, disc_cond_input_repeat])
    num_layers = int(np.log2(num_sample_inputs) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    for i in range(num_layers):
        disc_model = Conv1D(curr_conv_filters, filter_width,
                            strides=1, padding="same", kernel_regularizer=l2())(disc_model)
        if activation == "leaky":
            disc_model = LeakyReLU(0.2)(disc_model)
        else:
            disc_model = Activation(activation)(disc_model)
        disc_model = Dropout(dropout_alpha)(disc_model)
        disc_model = MaxPool1D()(disc_model)
        curr_conv_filters *= 2
    disc_model = Conv1D(curr_conv_filters, filter_width, padding="same", kernel_regularizer=l2())(disc_model)
    if activation == "leaky":
        disc_model = LeakyReLU(0.2)(disc_model)
    else:
        disc_model = Activation(activation)(disc_model)
    disc_model = Flatten()(disc_model)
    disc_model = Dense(1, kernel_regularizer=l2())(disc_model)
    disc_model = Activation("sigmoid")(disc_model)
    discriminator = Model([disc_cond_input, disc_sample_input], disc_model)
    return discriminator


def generator_dense(num_cond_inputs=1, num_random_inputs=1, num_hidden_layers=1, num_hidden_neurons=8, num_outputs=1,
                    dropout_alpha=0.2, activation="selu", l2_strength=0.01, normalize=True):
    """
    Dense conditional generator network. Includes 1 hidden layer.

    Args:
        num_cond_inputs (int): Size of the conditional input vector.
        num_random_inputs (int): Size of the random input vector.
        num_hidden (int): Number of hidden neurons
        num_outputs (int): Number of output neurons
        activation (str): Activation function for the hidden layer.

    Returns:
        Generator Keras Model object.
    """
    gen_cond_input = Input(shape=(num_cond_inputs, ))
    gen_rand_input = Input(shape=(num_random_inputs, ))
    gen_model = concatenate([gen_cond_input, gen_rand_input])
    for h in range(num_hidden_layers):
        gen_model = Dense(num_hidden_neurons, kernel_regularizer=l2(l2_strength))(gen_model)
        if activation == "leaky":
            gen_model = LeakyReLU(0.2)(gen_model)
        else:
            gen_model = Activation(activation)(gen_model)
        gen_model = Dropout(dropout_alpha)(gen_model)
    gen_model = Dense(num_outputs, kernel_regularizer=l2())(gen_model)
    gen_model = Reshape((num_outputs, 1))(gen_model)
    if normalize:
        gen_model = BatchNormalization()(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def discriminator_dense(num_cond_inputs=1, num_sample_inputs=1, num_hidden_neurons=8,
                        num_hidden_layers=2, activation="selu", l2_strength=0.01, dropout_alpha=0):
    """
    Dense conditional discriminator network.

    Args:
        num_cond_inputs (int): Size of the conditional input vector
        num_sample_inputs (int): Size of the sample input vector
        num_hidden_neurons (int): Number of hidden neurons
        num_hidden_layers (int): Number of hidden layers
        activation (str): Type of activation function
        l2_strength (float): Weight of l2 regularization on hidden layers
        dropout_alpha (float): Proportion (0 to 1) of previous inputs zeroed out
    Returns:
        Discriminator model
    """
    disc_cond_input = Input(shape=(num_cond_inputs, ))
    disc_sample_input = Input(shape=(num_sample_inputs, 1))
    disc_sample_flat = Flatten()(disc_sample_input)
    disc_model = concatenate([disc_cond_input, disc_sample_flat])
    for i in range(num_hidden_layers):
        disc_model = Dense(num_hidden_neurons, kernel_regularizer=l2(l2_strength))(disc_model)
        if activation == "leaky":
            disc_model = LeakyReLU(0.2)(disc_model)
        else:
            disc_model = Activation(activation)(disc_model)
        disc_model = Dropout(dropout_alpha)(disc_model)
        disc_model = GaussianNoise(0.1)(disc_model)
    disc_model = Dense(1, kernel_regularizer=l2(l2_strength))(disc_model)
    disc_model = Activation("sigmoid")(disc_model)
    discriminator = Model([disc_cond_input, disc_sample_input], disc_model)
    return discriminator


def stack_gen_disc(generator, discriminator):
    """
    Combines generator and discrminator layers together while freezing the weights of the discriminator layers.

    Args:
        generator (Keras Model object): Generator model
        discriminator (Keras Model object): Discriminator model

    Returns:
        Generator layers attached to discriminator layers.
    """
    discriminator.trainable = False
    model = discriminator([generator.input[0], generator.output])
    model_obj = Model(generator.input, model)
    return model_obj


def initialize_gan(generator, discriminator, loss, optimizer, metrics):
    """
    Compiles each of the GAN component models and stacks the generator and discrminator together

    Args:
        generator: Generator model object
        discriminator: Discriminator model object
        optimizer: Optimizer object or str referring to optimizer with default settings
        metrics: List of additional scoring metrics, such as accuracy

    Returns:
        Stacked Generator-discriminator model
    """
    generator.compile(optimizer=optimizer, loss=loss)
    discriminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    gen_disc = stack_gen_disc(generator, discriminator)
    gen_disc.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(generator.summary())
    print(discriminator.summary())
    print(gen_disc.summary())
    return gen_disc


def fit_condition_distributions(train_cond_data):
    """
    Calculate the scale parameter for the exponential distribution of correlated conditional variables
    for the Lorenz 96 model in time.

    Args:
        train_cond_data: array of conditioning values where the first column is the current X, and each
            other column is a lagged X value

    Returns:
        array of scale values
    """
    train_cond_exp_scale = np.zeros(train_cond_data.shape[1] - 1)
    for i in range(1, train_cond_data.shape[1]):
        train_cond_exp_scale[i - 1] = expon.fit(np.abs(train_cond_data[:, 0] - train_cond_data[:, i]), floc=0)[1]
    return train_cond_exp_scale


def generate_random_condition_data(batch_size, num_cond_inputs, train_cond_scale):
    """
    Generate correlated conditional random numbers to train the generator network.

    Args:
        batch_size: number of random samples
        num_cond_inputs: number of conditional inputs
        train_cond_scale: exponential distribution scale values

    Returns:

    """
    batch_cond_data = np.zeros((batch_size, num_cond_inputs, 1))
    batch_cond_data[:, 0, 0] = np.random.normal(size=batch_size)
    for t in range(1, train_cond_scale.size + 1):
        batch_cond_data[:, t , 0] = batch_cond_data[:, 0, 0] + \
                                    np.random.choice([-1, 1], size=batch_size) * expon.rvs(loc=0,
                                                                                           scale=train_cond_scale[t-1],
                                                                                           size=batch_size)
    return batch_cond_data


def train_gan(train_sample_data, train_cond_data, generator, discriminator, gen_disc,
              batch_size, random_vec_size, gan_path, gan_index, num_epochs=(1, 5, 10), metrics=("accuracy",),
              out_dtype="float32"):
    """
    Trains the full GAN model on provided training data.

    Args:
        train_sample_data: Training samples
        train_cond_data: Training condition vectors paired with samples
        generator: Generator object
        discriminator: Discriminator object
        gen_disc: Generator-discriminator stack object
        batch_size: Number of training examples per batch
        random_vec_size: Size of the random vector for the generator
        gan_path: Path to saving GAN model objects
        gan_index: GAN configuration index
        num_epochs: List of epochs to output saved models
        metrics: List of metrics for training
        sample_scaling_values: pandas dataframe of mean and standard deviation for training data
        cond_scaling_values: pandas dataframe of mean and standard deviation for conditional data
        out_dtype: Datatype of synthetic output

    Returns:

    """
    batch_size = int(batch_size)
    batch_half = int(batch_size // 2)
    train_order = np.arange(train_sample_data.shape[0])
    disc_loss_history = []
    gen_loss_history = []
    time_history = []
    current_epoch = []
    batch_labels = np.zeros(batch_size, dtype=int)
    batch_labels[:batch_half] = 1
    gen_labels = np.ones(batch_size, dtype=int)
    batch_vec = np.zeros((batch_size, random_vec_size))
    gen_batch_vec = np.zeros((batch_size, random_vec_size), dtype=train_sample_data.dtype)
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_sample_data.shape[1:]]), dtype=out_dtype)
    combo_cond_data_batch = np.zeros((batch_size, train_cond_data.shape[1]), dtype=out_dtype)
    gen_cond_data_batch = np.zeros((batch_size, train_cond_data.shape[1]), dtype=out_dtype)
    hist_cols = ["Epoch", "Batch", "Disc Loss"] + ["Disc " + m for m in metrics] + \
                ["Gen Loss"] + ["Gen " + m for m in metrics]
    # Loop over each epoch
    gen_pred_func = predict_stochastic(generator)
    for epoch in range(1, max(num_epochs) + 1):
        np.random.shuffle(train_order)
        # Loop over all of the random training batches
        for b, b_index in enumerate(np.arange(batch_size, train_sample_data.shape[0], batch_size * 2)):
            batch_vec[:] = np.random.normal(size=(batch_size, random_vec_size))
            gen_batch_vec[:] = np.random.normal(size=(batch_size, random_vec_size))
            gen_cond_data_batch[:] = train_cond_data[train_order[b_index: b_index + batch_size]]
            combo_cond_data_batch[:] = train_cond_data[train_order[b_index - batch_size: b_index]]
            combo_data_batch[:batch_half] = train_sample_data[train_order[b_index - batch_size: b_index - batch_half]]
            combo_data_batch[batch_half:] = gen_pred_func([combo_cond_data_batch[batch_half:],
                                                           batch_vec[batch_half:], True])[0]
            disc_loss_history.append(discriminator.train_on_batch([combo_cond_data_batch,
                                                                   combo_data_batch],
                                                                  batch_labels))
            gen_loss_history.append(gen_disc.train_on_batch([gen_cond_data_batch, gen_batch_vec],
                                                            gen_labels))
            print("Epoch {0:02d}, Batch {1:04d}, Disc Loss: {2:0.4f}, Gen Loss: {3:0.4f}".format(epoch,
                                                                                                 b,
                                                                                                 disc_loss_history[-1][0],
                                                                                                 gen_loss_history[-1][0]))
            time_history.append(pd.Timestamp("now"))
            current_epoch.append((epoch, b))
        if epoch in num_epochs:
            print("{2} Save Models Combo: {0} Epoch: {1}".format(gan_index,
                                                                 epoch,
                                                                 pd.Timestamp("now")))
            generator.save(join(gan_path, "gan_generator_{0:04d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            discriminator.save(join(gan_path, "gan_discriminator_{0:04d}_{1:04d}.h5".format(gan_index, epoch)))
            time_history_index = pd.DatetimeIndex(time_history)
            history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history,
                                              gen_loss_history]),
                                   index=time_history_index, columns=hist_cols)
            history.to_csv(join(gan_path, "gan_loss_history_{0:04d}.csv".format(gan_index)), index_label="Time")
    time_history_index = pd.DatetimeIndex(time_history)
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history,
                                      gen_loss_history]),
                           index=time_history_index, columns=hist_cols)
    history.to_csv(join(gan_path, "gan_loss_history_{0:04d}.csv".format(gan_index)), index_label="Time")
    return


def normalize_data(data, scaling_values=None):
    """
    Normalize each channel in the multi dimensional data matrix independently.

    Args:
        data: multi-dimensional array with dimensions (example, ..., channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
        for i in range(data.shape[-1]):
            scaling_values.loc[i, ["mean", "std"]] = [data[..., i].mean(), data[..., i].std()]
    for i in range(data.shape[-1]):
        normed_data[..., i] = (data[..., i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
    return normed_data, scaling_values


def unnormalize_data(normed_data, scaling_values):
    """
    Re-scale normalized data back to original values

    Args:
        normed_data: normalized data
        scaling_values: pandas dataframe of mean and standard deviation from normalize_data

    Returns:
        Re-scaled data
    """
    data = np.zeros(normed_data.shape, dtype=normed_data.dtype)
    for i in range(normed_data.shape[-1]):
        data[..., i] = normed_data[..., i] * scaling_values.loc[i, "std"] + scaling_values.loc[i, "mean"]
    return data


def main():
    gen_model = generator_conv()
    disc_model = discriminator_conv()
    metrics = ["accuracy"]
    opt = Adam(lr=0.0001, beta_1=0.5)
    gen_disc = initialize_gan(gen_model, disc_model, opt, metrics)
    return


if __name__ == "__main__":
    main()
