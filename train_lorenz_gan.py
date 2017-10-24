from lorenz_gan.lorenz import run_lorenz96_truth, l96_truth_step
from lorenz_gan.gan import generator_conv, generator_dense, discriminator_conv, discriminator_dense, stack_gen_disc
from lorenz_gan.gan import train_gan, initialize_gan, normalize_data, unnormalize_data
from keras.optimizers import Adam
import numpy as np
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", required=True, help="Config yaml file")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    X_out, Y_out = generate_lorenz_data(config["lorenz"])
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
    X_series_list = []
    for j in range(X_out.shape[1]):
        X_series_list.append(np.concatenate([X_out[i:i-cond_i:-1]
                               for i in range(cond_i, X_out.shape[0])]))
    X_series = np.concatenate(X_series_list)
    return

def train_lorenz_gan(config, X_out, Y_out):

    if config["gan"]["structure"] == "dense":
        gen_model = generator_dense(**config["gan"]["generator"])
        disc_model = discriminator_dense(**config["gan"]["discriminator"])
    else:
        gen_model = generator_conv(**config["gan"]["generator"])
        disc_model = discriminator_conv(**config["gan"]["discriminator"])
    optimizer = Adam(lr=0.0001, beta_1=0.5)
    gen_disc = initialize_gan(gen_model, disc_model, optimizer, config["gan"]["metrics"])
    train_gan()



if __name__ == "__main__":
    main()
