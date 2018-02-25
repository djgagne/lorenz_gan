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
    if type(initial_step) == int:
        initial_steps = np.array([initial_step], dtype=int)
    else:
        initial_steps = np.linspace(initial_step[0], initial_step[1], initial_step[2]).astype(int)
    out_path = config["out_path"]
    x_only = config["x_only"]
    if "call_param_once" in config.keys():
        call_param_once = config["call_param_once"]
    else:
        call_param_once = False
    f = config["F"]
    lorenz_output = xr.open_dataset(config["lorenz_nc_file"])
    step_values = lorenz_output["step"].values
    if args.proc == 1:
        for step in initial_steps:
            step_index = np.where(step_values == step)[0]
            print(step_index)
            x_initial = lorenz_output["lorenz_x"][step_index].values
            y_initial = lorenz_output["lorenz_y"][step_index].values
            u_initial = y_initial.reshape(8, 32).sum(axis=1)
            for member in range(config["num_members"]):
                launch_forecast_member(member, np.copy(x_initial), u_initial, f, u_model_path, random_updater,
                                       num_steps, num_random, time_step,
                                       random_seeds[member], initial_step, x_only, call_param_once, out_path)
    else:
        pool = Pool(args.proc)
        for step in initial_steps:
            step_index = np.where(step_values == step)[0][0]
            print(step_index)
            x_initial = lorenz_output["lorenz_x"][step_index].values
            y_initial = lorenz_output["lorenz_y"][step_index].values
            u_initial = y_initial.reshape(8, 32).sum(axis=1)
            print(x_initial, y_initial, u_initial)
            for member in range(config["num_members"]):
                pool.apply_async(launch_forecast_member, (member, np.copy(x_initial), u_initial, f, u_model_path,
                                                          random_updater, num_steps,
                                                          num_random, time_step,
                                                          random_seeds[member], initial_step, x_only, call_param_once,
                                                          out_path))
        pool.close()
        pool.join()
    return


def launch_forecast_member(member_number, x_initial, u_initial, f, u_model_path, random_updater, num_steps,
                           num_random, time_step, random_seed, initial_step, x_only, call_param_once, out_path):
    """
    Run a single Lorenz 96 forecast model with a specified x and u initial conditions and forcing. The output
    of the run is saved to a netCDF file.

    Args:
        member_number:
        x_initial:
        u_initial:
        f:
        u_model_path:
        random_updater:
        num_steps:
        num_random:
        time_step:
        random_seed:
        initial_step:
        x_only:
        call_param_once:
        out_path:

    Returns:

    """
    try:
        sess = K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1,
                                                    inter_op_parallelism_threads=1))
        K.set_session(sess)
        if u_model_path[-2:] == "h5":
            u_model = SubModelGAN(u_model_path)
        else:
            with open(u_model_path, "rb") as u_model_file:
                u_model = pickle.load(u_model_file)
        print("Starting member {0:d}".format(member_number))
        np.random.seed(random_seed)
        K.tf.set_random_seed(random_seed)
        forecast_out = run_lorenz96_forecast(x_initial, u_initial, f, u_model, random_updater, num_steps, num_random,
                                             time_step, x_only=x_only, call_param_once=call_param_once)
        forecast_out.attrs["initial_step"] = initial_step
        forecast_out.attrs["member"] = member_number
        forecast_out.attrs["u_model_path"] = u_model_path
        forecast_out.to_netcdf(join(out_path, "lorenz_forecast_{0:07d}_{1:02d}.nc".format(initial_step, member_number)),
                               mode="w", encoding={"x": {"dtype": "float32", "zlib": True, "complevel": 2},
                                                   "u": {"dtype": "float32", "zlib": True, "complevel": 2}})
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


if __name__ == "__main__":
    main()