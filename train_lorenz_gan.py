from lorenz_gan.lorenz import run_lorenz96_truth, process_lorenz_data, save_lorenz_output
from lorenz_gan.gan import generator_conv, generator_dense, discriminator_conv, discriminator_dense
from lorenz_gan.gan import predict_stochastic, generator_dense_stoch, discriminator_conv_concrete, generator_dense_auto_stoch
from lorenz_gan.gan import train_gan, initialize_gan, normalize_data, generator_conv_concrete, unnormalize_data
from lorenz_gan.submodels import AR1RandomUpdater, SubModelHist, SubModelPoly, SubModelPolyAdd, SubModelANNRes, SubModelANN
import tensorflow as tf
import xarray as xr
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import pickle
import pandas as pd
import yaml
import argparse
from os.path import exists, join
from os import mkdir


def main():
    """
    This script runs the Lorenz '96 model and then trains a generative adversarial network
    to parameterize the unresolved Y values. The script requires a config file as input.
    The config file is formatted in the yaml format with the following information included.

    lorenz: # The Lorenz model subsection
        K: 8 # number of X variables
        J: 32 # number of Y variables per X variable
        h: 1 # coupling constant
        b: 10 # spatial-scale ratio
        c: 10 # time scale ratio
        F: 30 # forcing term
        time_step: 0.001 # time step of Lorenz truth model in MTU
        num_steps: 1000000 # number of integration time steps
        skip: 5 # number of steps to skip when saving out the model
        burn_in: 2000 # number of steps to remove from the beginning of the integration
    gan: # The GAN subsection
        structure: conv # type of GAN neural network, options are conv or dense
        t_skip: 10 # number of time steps to skip when saving data for training
        x_skip: 1 # number of X variables to skip
        output: sample # Train the neural network to output a "sample" of Ys or the "mean" of the Ys
        generator:
            num_cond_inputs: 3 # number of conditional X values
            num_random_inputs: 13 # number of random values
            num_outputs: 32 # number of output variables (should match J)
            activation: relu # activation function
            min_conv_filters: 32 # number of convolution filters in the last layer of the generator
            min_data_width: 4 # width of the data array after the dense layer in the generator
            filter_width: 4 # Size of the convolution filters
        discriminator:
            num_cond_inputs: 3 # number of conditional X values
            num_sample_inputs: 32 # number of Y values
            activation: relu # Activation function
            min_conv_filters: 32 # number of convolution filters in the first layer of the discriminator
            min_data_width: 4 # width of the data array before the dense layer in the discriminator
            filter_width: 4 # width of the convolution filters
        gan_path: ./exp # path where GAN files are saved
        batch_size: 64 # Number of examples per training batch
        gan_index: 0 # GAN configuration number
        loss: binary_crossentropy # Loss function for the GAN
        num_epochs: [1, 5, 10] # Epochs after which the GAN model is saved
        metrics: ["accuracy"] # Metrics to calculate along with the loss
    output_nc_file: ./exp/lorenz_output.nc # Where Lorenz 96 data is output
    output_csv_file: ./exp/lorenz_combined_output.csv # Where flat file formatted data is saved

    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="lorenz.yaml", help="Config yaml file")
    parser.add_argument("-r", "--reload", action="store_true", default=False, help="Reload netCDF and csv files")
    parser.add_argument("-g", "--gan", action="store_true", default=False, help="Train GAN")

    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    if not exists(config["gan"]["gan_path"]):
        mkdir(config["gan"]["gan_path"])
    u_scale = config["lorenz"]["h"] * config["lorenz"]["c"] / config["lorenz"]["b"]
    saved_steps = (config["lorenz"]["num_steps"] - config["lorenz"]["burn_in"]) // config["lorenz"]["skip"]
    split_step = int(config["lorenz"]["train_test_split"] * saved_steps)
    #val_split_step = int(config["lorenz"]["val_split"] * saved_steps)

    if args.reload:
        print("Reloading csv data")
        combined_data = pd.read_csv(config["output_csv_file"])
        lorenz_output = xr.open_dataset(config["output_nc_file"])
        X_out = lorenz_output["lorenz_x"].values
    else:
        X_out, Y_out, times, steps = generate_lorenz_data(config["lorenz"])
        print(X_out.shape, Y_out.shape, saved_steps, split_step)
        combined_data = process_lorenz_data(X_out[:split_step], times[:split_step],
                                            steps[:split_step],
                                            config["lorenz"]["J"], config["lorenz"]["F"],
                                            config["lorenz"]["time_step"] * config["lorenz"]["skip"],
                                            config["gan"]["x_skip"],
                                            config["gan"]["t_skip"], u_scale)
        combined_test_data = process_lorenz_data(X_out[split_step:], times[split_step:],
                                                 steps[split_step:],
                                                 config["lorenz"]["J"], config["lorenz"]["F"],
                                                 config["lorenz"]["time_step"] * config["lorenz"]["skip"],
                                                 config["gan"]["x_skip"],
                                                 config["gan"]["t_skip"], u_scale)
        save_lorenz_output(X_out, Y_out, times, steps, config["lorenz"], config["output_nc_file"])
        combined_data.to_csv(config["output_csv_file"], index=False)
        combined_test_data.to_csv(str(config["output_csv_file"]).replace(".csv", "_test.csv"))
    train_random_updater(X_out[:, 1], config["random_updater"]["out_file"])
    u_vals = combined_data["u_scale"] * combined_data["Ux_t+1"]
    train_histogram(combined_data["X_t"].values,
                    u_vals, **config["histogram"])
    train_poly(combined_data["X_t"].values, u_vals, **config["poly"])
    x_time_series = X_out[:split_step-1, 0:1]
    u_time_series = (-X_out[:split_step-1, -1] * (X_out[:split_step-1, -2] - X_out[:split_step-1, 1])
                     - X_out[:split_step-1, 0] + config["lorenz"]["F"]) \
        - (X_out[1:split_step, 0] - X_out[:split_step-1, 0]) / config["lorenz"]["time_step"] / config["lorenz"]["skip"]
    #x_val_time_series = X_out[split_step:val_split_step - 1, 0:1]
    #u_val_time_series = (-X_out[split_step:val_split_step - 1, -1] * (X_out[split_step:val_split_step - 1, -2] - X_out[split_step:val_split_step - 1, 1])
    #                 - X_out[split_step:val_split_step - 1, 0] + config["lorenz"]["F"]) \
    #                - (X_out[split_step + 1:val_split_step, 0] - X_out[split_step:val_split_step - 1, 0]) / config["lorenz"]["time_step"] / \
    #                config["lorenz"]["skip"]
    combined_time_series = pd.DataFrame({"X_t": x_time_series[1:].ravel(), "Ux_t": u_time_series[:-1],
                                         "Ux_t+1": u_time_series[1:]}, columns=["X_t", "Ux_t", "Ux_t+1"])
    print(u_time_series.min(), u_time_series.max(), u_time_series.mean())
    combined_time_series.to_csv(config["output_csv_file"].replace(".csv", "_ts_val.csv"))
    if "poly_add" in config.keys():
        train_poly_add(x_time_series,
                       u_time_series,
                       **config["poly_add"])
    if "ann" in config.keys():
        print("X in", x_time_series.min(), x_time_series.max())
        print("U out", u_time_series.min(), u_time_series.max())
        train_ann(x_time_series,
                      u_time_series,
                      config["ann"])
    if "ann_res" in config.keys():
        print("X in", x_time_series.min(), x_time_series.max())
        print("U out", u_time_series.min(), u_time_series.max())

        train_ann_res(x_time_series,
                      u_time_series,
                      config["ann_res"])
    if args.gan:
        train_lorenz_gan(config, combined_data, combined_time_series)
    return


def generate_lorenz_data(config):
    """
    Run the Lorenz '96 truth model

    Args:
        config:

    Returns:

    """
    x = np.zeros(config["K"], dtype=np.float32)
    # initialize Y array
    y = np.zeros(config["J"] * config["K"], dtype=np.float32)
    x[0] = 1
    y[0] = 1
    skip = config["skip"]
    x_out, y_out, times, steps = run_lorenz96_truth(x, y, config["h"], config["F"], config["b"],
                                                    config["c"], config["time_step"], config["num_steps"],
                                                    config["burn_in"], skip)
    return x_out, y_out, times, steps


def train_lorenz_gan(config, combined_data, combined_time_series):
    """
    Train GAN on Lorenz data

    Args:
        config:
        combined_data:

    Returns:

    """
    if "num_procs" in config.keys():
        num_procs = config["num_procs"]
    else:
        num_procs = 1
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_procs,
                                                inter_op_parallelism_threads=1))
    K.set_session(sess)
    x_cols = config["gan"]["cond_inputs"]
    y_cols = config["gan"]["output_cols"]
    X_series = combined_data[x_cols].values
    Y_series = combined_data[y_cols].values
    X_norm, X_scaling_values = normalize_data(X_series)
    if config["gan"]["output"].lower() == "mean":
        Y_norm, Y_scaling_values = normalize_data(np.expand_dims(Y_series.mean(axis=1), axis=-1))
    else:
        Y_norm, Y_scaling_values = normalize_data(Y_series)
    X_scaling_values.to_csv(join(config["gan"]["gan_path"],
                                 "gan_X_scaling_values_{0:04d}.csv".format(config["gan"]["gan_index"])),
                            index_label="Channel")
    Y_scaling_values.to_csv(join(config["gan"]["gan_path"],
                                 "gan_Y_scaling_values_{0:04d}.csv".format(config["gan"]["gan_index"])),
                            index_label="Channel")
    trim = X_norm.shape[0] % config["gan"]["batch_size"]
    if config["gan"]["structure"] == "dense":
        gen_model = generator_dense(**config["gan"]["generator"])
        disc_model = discriminator_dense(**config["gan"]["discriminator"])
        rand_vec_length = config["gan"]["generator"]["num_random_inputs"]
    elif config["gan"]["structure"] == "specified_random":
        gen_model = generator_dense_stoch(**config["gan"]["generator"])
        disc_model = discriminator_dense(**config["gan"]["discriminator"])
        rand_vec_length = config["gan"]["generator"]["num_random_inputs"] + \
                          2 * config["gan"]["generator"]["num_hidden_neurons"] + \
                          config["gan"]["generator"]["num_cond_inputs"]
    elif config["gan"]["structure"] == "auto_stoch":
        gen_model = generator_dense_auto_stoch(**config["gan"]["generator"])
        disc_model = discriminator_dense(**config["gan"]["discriminator"])
        rand_vec_length = config["gan"]["generator"]["num_random_inputs"] + \
                          2 * config["gan"]["generator"]["num_hidden_neurons"] + \
                          config["gan"]["generator"]["num_cond_inputs"]
    elif config["gan"]["structure"] == "concrete":
        gen_model = generator_conv_concrete(**config["gan"]["generator"])
        disc_model = discriminator_conv_concrete(**config["gan"]["discriminator"])
        rand_vec_length = config["gan"]["generator"]["num_random_inputs"]

    else:
        gen_model = generator_conv(**config["gan"]["generator"])
        disc_model = discriminator_conv(**config["gan"]["discriminator"])
        rand_vec_length = config["gan"]["generator"]["num_random_inputs"]
    optimizer = Adam(lr=config["gan"]["learning_rate"], beta_1=0.5, beta_2=0.9)
    loss = config["gan"]["loss"]
    gen_disc = initialize_gan(gen_model, disc_model, loss, optimizer, config["gan"]["metrics"])
    if trim > 0:
        Y_norm = Y_norm[:-trim]
        X_norm = X_norm[:-trim]
    train_gan(np.expand_dims(Y_norm, -1), X_norm, gen_model, disc_model, gen_disc, config["gan"]["batch_size"],
              rand_vec_length, config["gan"]["gan_path"],
              config["gan"]["gan_index"], config["gan"]["num_epochs"], config["gan"]["metrics"])
    gen_pred_func = predict_stochastic(gen_model)
    x_ts_norm, _ = normalize_data(combined_time_series[x_cols].values,
                                scaling_values=X_scaling_values)
    gen_ts_pred_norm = gen_pred_func([x_ts_norm,
                                         np.zeros((x_ts_norm.shape[0], rand_vec_length)), 0])[0]
    print(gen_ts_pred_norm.shape)
    gen_ts_preds = unnormalize_data(gen_ts_pred_norm, scaling_values=Y_scaling_values)
    gen_ts_residuals = combined_time_series[y_cols].values.ravel() - gen_ts_preds.ravel()
    train_random_updater(gen_ts_residuals,
                         config["random_updater"]["out_file"].replace(".pkl",
                                                                      "_{0:04d}.pkl".format(config["gan"]["gan_index"])))


def train_random_updater(data, out_file):
    random_updater = AR1RandomUpdater()
    random_updater.fit(data)
    print("AR1 Corr:", random_updater.corr)
    print("AR1 Noise SD:", random_updater.noise_sd)
    with open(out_file, "wb") as out_file_obj:
        pickle.dump(random_updater, out_file_obj, pickle.HIGHEST_PROTOCOL)


def train_histogram(x_data, u_data, num_x_bins=10, num_u_bins=10, out_file="./histogram.pkl"):
    hist_model = SubModelHist(num_x_bins, num_u_bins)
    hist_model.fit(x_data, u_data)
    with open(out_file, "wb") as out_file_obj:
        pickle.dump(hist_model, out_file_obj, pickle.HIGHEST_PROTOCOL)


def train_poly(x_data, u_data, num_terms=3, noise_type="additive", out_file="./poly.pkl"):
    poly_model = SubModelPoly(num_terms=num_terms, noise_type=noise_type)
    poly_model.fit(x_data, u_data)
    with open(out_file, "wb") as out_file_obj:
        pickle.dump(poly_model, out_file_obj, pickle.HIGHEST_PROTOCOL)
    return


def train_poly_add(x_data, u_data, num_terms=3, out_file="./poly_add.pkl"):
    poly_add_model = SubModelPolyAdd(num_terms=num_terms)
    poly_add_model.fit(x_data, u_data)
    with open(out_file, "wb") as out_file_obj:
        pickle.dump(poly_add_model, out_file_obj, pickle.HIGHEST_PROTOCOL)


def train_ann(x_data, u_data, config):
    print("ANN Input shapes", x_data.shape, u_data.shape)
    ann_model = SubModelANN(**config)
    ann_model.fit(x_data, u_data)
    ann_model.save_model(config["out_path"])

def train_ann_res(x_data, u_data, config):
    ann_res_model = SubModelANNRes(**config)
    ann_res_model.fit(x_data, u_data)
    ann_res_model.save_model(config["out_path"])


if __name__ == "__main__":
    main()
