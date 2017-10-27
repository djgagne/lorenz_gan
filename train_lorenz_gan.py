from lorenz_gan.lorenz import run_lorenz96_truth, process_lorenz_data, save_lorenz_output
from lorenz_gan.gan import generator_conv, generator_dense, discriminator_conv, discriminator_dense
from lorenz_gan.gan import train_gan, initialize_gan, normalize_data
from keras.optimizers import Adam
import numpy as np
import yaml
import argparse
from os.path import exists, join
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
    X_out, Y_out, times, steps = generate_lorenz_data(config["lorenz"])
    combined_data = process_lorenz_data(X_out, Y_out, times, steps,
                                        config["gan"]["generator"]["num_cond_inputs"],
                                        config["lorenz"]["J"], config["gan"]["x_skip"],
                                        config["gan"]["t_skip"])
    save_lorenz_output(X_out, Y_out, times, steps, config["lorenz"], config["output_nc_file"])
    combined_data.to_csv(config["output_csv_file"], index=False)
    print(combined_data)
    train_lorenz_gan(config, combined_data)
    return


def generate_lorenz_data(config):
    X = np.zeros(config["K"], dtype=np.float32)
    Y = np.zeros(config["J"] * config["K"], dtype=np.float32)
    X[0] = 1
    Y[0] = 1
    skip = config["skip"]
    X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, config["h"], config["F"], config["b"],
                                                    config["c"], config["time_step"], config["num_steps"])
    return (X_out[config['burn_in']::skip], Y_out[config["burn_in"]::skip],
            times[config["burn_in"]::skip], steps[config["burn_in"]::skip])


def train_lorenz_gan(config, combined_data):
    X_series = np.expand_dims(combined_data[["X_t", "X_t-1", "X_t-2"]].values, axis=-1)
    Y_series = np.expand_dims(combined_data[["Y_{0:d}".format(y) for y in range(config["lorenz"]["J"])]].values,
                              axis=-1)
    X_norm, X_scaling_values = normalize_data(X_series)
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
    else:
        gen_model = generator_conv(**config["gan"]["generator"])
        disc_model = discriminator_conv(**config["gan"]["discriminator"])
    optimizer = Adam(lr=0.0001, beta_1=0.5)
    loss = config["gan"]["loss"]
    gen_disc = initialize_gan(gen_model, disc_model, loss, optimizer, config["gan"]["metrics"])
    if trim > 0:
        Y_norm = Y_norm[:-trim]
        X_norm = X_norm[:-trim]
    train_gan(Y_norm, X_norm, gen_model, disc_model, gen_disc, config["gan"]["batch_size"],
              config["gan"]["generator"]["num_random_inputs"], config["gan"]["gan_path"],
              config["gan"]["gan_index"], config["gan"]["num_epochs"], config["gan"]["metrics"],
              Y_scaling_values, X_scaling_values)


if __name__ == "__main__":
    main()
