import numpy as np
import pandas as pd
import xarray as xr
import yaml
import argparse
from lorenz_gan.lorenz import run_lorenz96_forecast
from lorenz_gan.submodels import SubModelGAN
from multiprocessing import Pool
import pickle
import traceback
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    u_model_path = config["u_model_path"]
    with open(config["random_updater_path"], "rb") as random_updater_file:
        random_updater = pickle.load(random_updater_file)
    num_steps = config["num_steps"]
    num_random = config["num_random"]
    time_step = config["time_step"]
    x_time_lag = config["x_time_lag"]
    random_seeds = config["random_seeds"]
    initial_step = config["initial_step"]
    out_path = config["out_path"]
    F = config["F"]
    lorenz_output = xr.open_dataset(config["lorenz_nc_file"])
    x_initial = lorenz_output["lorenz_x"][initial_step].values
    if args.proc == 1:
        for member in range(config["num_members"]):
            launch_forecast_member(member, np.copy(x_initial), F, u_model_path, random_updater, num_steps, num_random, time_step,
                                   x_time_lag, random_seeds[member], out_path)
    else:
        pool = Pool(args.proc)
        for member in range(config["num_members"]):
            pool.apply_async(launch_forecast_member, (member, np.copy(x_initial), F, u_model_path, random_updater, num_steps,
                                                      num_random, time_step,
                                                      x_time_lag, random_seeds[member], out_path))
        pool.close()
        pool.join()
    return


def launch_forecast_member(member_number, x_initial, F, u_model_path, random_updater, num_steps, num_random, time_step,
                           x_time_lag, random_seed, out_path):
    try:
        if u_model_path[-2:] == "h5":
            u_model = SubModelGAN(u_model_path)
        else:
            with open(u_model_path, "rb") as u_model_file:
                u_model = pickle.load(u_model_file)
                print(u_model.histogram)
        print("Starting member {0:d}".format(member_number))
        np.random.seed(random_seed)
        x_out, times, steps = run_lorenz96_forecast(x_initial,
                                                    F, u_model, random_updater, num_steps, num_random,
                                                    time_step, x_time_lag)
        x_data = {"time": times, "step": steps}
        x_cols = []
        for i in range(x_out.shape[1]):
            x_cols.append("X_{0:d}".format(i))
            x_data[x_cols[-1]] = x_out[:, i]
        x_frame = pd.DataFrame(x_data, columns=["time", "step"] + x_cols)
        x_frame.to_csv(join(out_path, "lorenz_forecast_{0:02d}.csv".format(member_number)), index=False)
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


if __name__ == "__main__":
    main()