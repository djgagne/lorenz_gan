import numpy as np
import xarray as xr
import yaml
import argparse
import keras.backend as K
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
    if len(config["random_seeds"]) != config["num_members"]:
        random_seeds = np.linspace(config["random_seeds"][0],
                                   config["random_seeds"][1],
                                   config["num_members"]).astype(int)
    else:
        random_seeds = config["random_seeds"]
    initial_step = config["initial_step"]
    out_path = config["out_path"]
    x_only = config["x_only"]
    F = config["F"]
    lorenz_output = xr.open_dataset(config["lorenz_nc_file"])
    x_initial = lorenz_output["lorenz_x"][initial_step].values
    y_initial = lorenz_output["lorenz_y"][initial_step].values
    u_initial = y_initial.reshape(8, 32).sum(axis=1)

    if args.proc == 1:
        for member in range(config["num_members"]):
            launch_forecast_member(member, np.copy(x_initial), u_initial, F, u_model_path, random_updater,
                                   num_steps, num_random, time_step,
                                   random_seeds[member], initial_step, x_only, out_path)
    else:
        pool = Pool(args.proc)
        for member in range(config["num_members"]):
            pool.apply_async(launch_forecast_member, (member, np.copy(x_initial), u_initial, F, u_model_path,
                                                      random_updater, num_steps,
                                                      num_random, time_step,
                                                      random_seeds[member], initial_step, x_only, out_path))
        pool.close()
        pool.join()
    return


def launch_forecast_member(member_number, x_initial, u_initial, f, u_model_path, random_updater, num_steps,
                           num_random, time_step, random_seed, initial_step, x_only, out_path):
    try:
        if u_model_path[-2:] == "h5":
            u_model = SubModelGAN(u_model_path)
        else:
            with open(u_model_path, "rb") as u_model_file:
                u_model = pickle.load(u_model_file)
        print("Starting member {0:d}".format(member_number))
        np.random.seed(random_seed)
        K.tf.set_random_seed(random_seed)
        forecast_out = run_lorenz96_forecast(x_initial, u_initial, f, u_model, random_updater, num_steps, num_random,
                                             time_step, x_only=x_only)
        forecast_out.attrs["initial_step"] = initial_step
        forecast_out.attrs["member"] = member_number
        forecast_out.to_netcdf(join(out_path, "lorenz_forecast_{0:07d}_{1:02d}.nc".format(initial_step, member_number)),
                               mode="w", encoding={"x": {"dtype": "float32", "zlib": True, "complevel": 2},
                                                   "u": {"dtype": "float32", "zlib": True, "complevel": 2}})
        #x_data = {"time": times, "step": steps}
        #x_cols = []
        #for i in range(x_out.shape[1]):
        #    x_cols.append("X_{0:d}".format(i))
        #    x_data[x_cols[-1]] = x_out[:, i]
        #x_frame = pd.DataFrame(x_data, columns=["time", "step"] + x_cols)
        #x_frame.to_csv(join(out_path, "lorenz_forecast_{0:02d}.csv".format(member_number)), index=False)
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


if __name__ == "__main__":
    main()