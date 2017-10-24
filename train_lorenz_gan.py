from lorenz_gan.lorenz import run_lorenz96_truth
from lorenz_gan.gan import generator_conv, generator_dense, discriminator_conv, discriminator_dense, stack_gen_disc
from lorenz_gan.gan import train_gan, initialize_gan, normalize_data, unnormalize_data
from keras.optimizers import Adam
import numpy as np
import yaml
import argparse
from os.path import exists
from os import mkdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="lorenz.yaml", help="Config yaml file")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    if not exists(config["gan"]["gan_path"]):
        mkdir(config["gan"]["gan_path"])
    X_out, Y_out = generate_lorenz_data(config["lorenz"])
    X_series, Y_series = process_lorenz_data(config, X_out, Y_out)
    print(X_series.shape)
    print(Y_series.shape)
    train_lorenz_gan(config, X_series, Y_series)
    return


def generate_lorenz_data(config):
    X = np.zeros(config["K"], dtype=np.float32)
    Y = np.zeros(config["J"] * config["K"], dtype=np.float32)
    X[0] = 1
    Y[0] = 1
    X_out, Y_out = run_lorenz96_truth(X, Y, config["h"], config["F"], config["b"],
                                      config["c"], config["time_step"], config["num_steps"])
    return X_out[config['burn_in']:], Y_out[config["burn_in"]:]


def process_lorenz_data(config, X_out, Y_out):
    cond_i = config["gan"]["generator"]["num_cond_inputs"]
    J = config["lorenz"]["J"]
    X_series_list = []
    Y_series_list = []
    x_skip = config["gan"]["x_skip"]
    t_skip = config["gan"]["t_skip"]
    for k in range(0, X_out.shape[1], x_skip):
        X_series_list.append(np.stack([X_out[i:i-cond_i:-1, k]
                             for i in range(cond_i, X_out.shape[0], t_skip)], axis=0))
        Y_series_list.append(Y_out[cond_i::t_skip, k * J: (k+1) * J])
    X_series = np.expand_dims(np.vstack(X_series_list), axis=-1)
    Y_series = np.expand_dims(np.vstack(Y_series_list), axis=-1)
    return X_series, Y_series


def train_lorenz_gan(config, X_series, Y_series):
    X_norm, X_scaling_values = normalize_data(X_series)
    Y_norm, Y_scaling_values = normalize_data(Y_series)
    trim = X_norm.shape[0] % config["gan"]["batch_size"]
    if config["gan"]["structure"] == "dense":
        gen_model = generator_dense(**config["gan"]["generator"])
        disc_model = discriminator_dense(**config["gan"]["discriminator"])
    else:
        gen_model = generator_conv(**config["gan"]["generator"])
        disc_model = discriminator_conv(**config["gan"]["discriminator"])
    optimizer = Adam(lr=0.0001, beta_1=0.5)
    gen_disc = initialize_gan(gen_model, disc_model, optimizer, config["gan"]["metrics"])
    train_gan(Y_norm[:-trim], X_norm[:-trim], gen_model, disc_model, gen_disc, config["gan"]["batch_size"],
              config["gan"]["generator"]["num_random_inputs"], config["gan"]["gan_path"],
              config["gan"]["gan_index"], config["gan"]["num_epochs"], config["gan"]["metrics"],
              Y_scaling_values, X_scaling_values)



if __name__ == "__main__":
    main()
