from keras.layers import concatenate, RepeatVector, Input, Flatten, BatchNormalization
from keras.layers import Conv1D, UpSampling1D, Dense, Activation, Reshape
from keras.optimizers import Adam
import xarray as xr
from scipy.stats import expon
from keras.models import Model
import numpy as np
import pandas as pd
from os.path import join


def generator_conv(num_cond_inputs=3, num_random_inputs=10, num_outputs=32,
                   activation="selu", min_conv_filters=8, min_data_width=4, filter_width=3):
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
    num_layers = int(np.log2(num_outputs) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers - 1))
    curr_conv_filters = max_conv_filters
    gen_cond_input = Input(shape=(num_cond_inputs, ))
    gen_rand_input = Input(shape=(num_random_inputs, ))
    gen_model = concatenate([gen_cond_input, gen_rand_input])
    gen_model = Dense(min_data_width * max_conv_filters)(gen_model)
    gen_model = Activation(activation)(gen_model)
    gen_model = Reshape((min_data_width, max_conv_filters))(gen_model)
    for i in range(0, num_layers):
        curr_conv_filters //= 2
        gen_model = Conv1D(curr_conv_filters, filter_width, padding="same")(gen_model)
        gen_model = Activation(activation)(gen_model)
        gen_model = UpSampling1D(size=2)(gen_model)
        gen_model = BatchNormalization()(gen_model)
    gen_model = Conv1D(1, filter_width, padding="same")(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def discriminator_conv(num_cond_inputs=3, num_sample_inputs=32, activation="selu",
                       min_conv_filters=8, min_data_width=4,
                       filter_width=3):
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
        disc_model = Conv1D(curr_conv_filters, filter_width, strides=2, padding="same")(disc_model)
        disc_model = Activation(activation)(disc_model)
        curr_conv_filters *= 2
    disc_model = Flatten()(disc_model)
    disc_model = Dense(1)(disc_model)
    disc_model = Activation("sigmoid")(disc_model)
    discriminator = Model([disc_cond_input, disc_sample_input], disc_model)
    return discriminator


def generator_dense(num_cond_inputs=3, num_random_inputs=10, num_hidden=20, num_outputs=32, activation="selu"):
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
    gen_model = Dense(num_hidden)(gen_model)
    gen_model = Activation(activation)(gen_model)
    gen_model = Dense(num_outputs)(gen_model)
    gen_model = Reshape((num_outputs, 1))(gen_model)
    generator = Model([gen_cond_input, gen_rand_input], gen_model)
    return generator


def discriminator_dense(num_cond_inputs=3, num_sample_inputs=32, num_hidden=20, activation="selu"):
    """
    Dense conditional discriminator network.

    Args:
        num_cond_inputs (int): Size of the conditional input vector
        num_sample_inputs (int): Size of the sample input vector
        num_hidden (int): Number of hidden neurons
        activation (str): Type of activation function

    Returns:
        Discriminator model
    """
    disc_cond_input = Input(shape=(num_cond_inputs, ))
    disc_sample_input = Input(shape=(num_sample_inputs, 1))
    disc_sample_flat = Flatten()(disc_sample_input)
    disc_model = concatenate([disc_cond_input, disc_sample_flat])
    disc_model = Dense(num_hidden)(disc_model)
    disc_model = Activation(activation)(disc_model)
    disc_model = Dense(1)(disc_model)
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
              sample_scaling_values=None, cond_scaling_values=None, out_dtype="float32", epoch_sample_size=1000):
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
    combo_cond_data_batch = np.zeros((batch_size, train_cond_data.shape[1], 1), dtype=out_dtype)
    train_cond_exp_scale = fit_condition_distributions(train_cond_data)
    gen_cond_data_batch = np.zeros((batch_size, train_cond_data.shape[1], 1), dtype=out_dtype)
    hist_cols = ["Epoch", "Batch", "Disc Loss"] + ["Disc " + m for m in metrics] + \
                ["Gen Loss"] + ["Gen " + m for m in metrics]
    # Loop over each epoch
    for epoch in range(1, max(num_epochs) + 1):
        np.random.shuffle(train_order)
        # Loop over all of the random training batches
        for b, b_index in enumerate(np.arange(batch_half, train_sample_data.shape[0] + batch_half, batch_half)):
            batch_vec[:] = np.random.normal(size=(batch_size, random_vec_size))
            gen_batch_vec[:] = np.random.normal(size=(batch_size, random_vec_size))
            gen_cond_data_batch[:] = generate_random_condition_data(batch_size, train_cond_data.shape[1],
                                                                    train_cond_exp_scale)
            combo_data_batch[:batch_half] = train_sample_data[train_order[b_index - batch_half: b_index]]
            combo_cond_data_batch[:batch_half] = train_cond_data[train_order[b_index - batch_half: b_index]]
            combo_cond_data_batch[batch_half:] = generate_random_condition_data(batch_half, train_cond_data.shape[1],
                                                                                train_cond_exp_scale)
            combo_data_batch[batch_half:] = generator.predict_on_batch([combo_cond_data_batch[batch_half:, :, 0],
                                                                        batch_vec[batch_half:]])
            disc_loss_history.append(discriminator.train_on_batch([combo_cond_data_batch[:, :, 0],
                                                                   combo_data_batch],
                                                                  batch_labels))
            gen_loss_history.append(gen_disc.train_on_batch([gen_cond_data_batch[:, :, 0], gen_batch_vec],
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
            gen_noise = np.random.normal(size=(epoch_sample_size, random_vec_size))
            gen_cond_noise = generate_random_condition_data(epoch_sample_size, train_cond_data.shape[1],
                                                            train_cond_exp_scale)
            gen_data_epoch = unnormalize_data(generator.predict([gen_cond_noise[:, :, 0], gen_noise]),
                                              sample_scaling_values)
            cond_data_epoch = unnormalize_data(gen_cond_noise, cond_scaling_values)
            gen_da = {}
            gen_da["gen_samples"] = xr.DataArray(gen_data_epoch.astype(out_dtype),
                                               coords={"t": np.arange(gen_data_epoch.shape[0]),
                                                       "y": np.arange(gen_data_epoch.shape[1]),
                                                        "channel": np.array([0])},
                                               dims=("t", "y", "channel"),
                                               attrs={"long_name": "Synthetic data", "units": ""})
            gen_da["gen_cond"] = xr.DataArray(cond_data_epoch.astype(out_dtype),
                                              coords={"t": np.arange(cond_data_epoch.shape[0]),
                                                      "c": np.arange(cond_data_epoch.shape[1]),
                                                      "channel": np.array([0])},
                                              dims=("t", "c", "channel"))
            xr.Dataset(gen_da).to_netcdf(join(gan_path,
                                              "gan_gen_patches_{0:04d}_epoch_{1:04d}.nc".format(gan_index, epoch)),
                                         encoding={"gen_samples": {"zlib": True,
                                                                   "complevel": 1},
                                                   "gen_cond": {"zlib": True,
                                                                "complevel": 1}})
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
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: 4-dimensional array with dimensions (example, y, x, channel/variable)
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
        scaling_values.loc[i, ["mean", "std"]] = [data[:, :, i].mean(), data[:, :, i].std()]
        normed_data[:, :, i] = (data[:, :, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
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
        data[:, :, i] = normed_data[:, :, i] * scaling_values.loc[i, "std"] + scaling_values.loc[i, "mean"]
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
