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
from os.path import join, exists
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    u_model_path = config["u_model_path"]
    random_updater_path = config["random_updater_path"]
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
        initial_steps = np.arange(initial_step[0], initial_step[1] + initial_step[2], initial_step[2]).astype(int)
    print("Initial steps", initial_steps, initial_steps.size)
    out_path = config["out_path"]
    x_only = config["x_only"]
    if "call_param_once" in config.keys():
        call_param_once = config["call_param_once"]
    else:
        call_param_once = False
    f = config["F"]
    lorenz_output = xr.open_dataset(config["lorenz_nc_file"])
    step_values = lorenz_output["step"].values
    members = np.arange(0, config["num_members"])
    if "num_tf_threads" in config.keys():
        num_tf_threads = config["num_tf_threads"]
    else:
        num_tf_threads = 1
    if args.proc == 1:
        for step in initial_steps:
            step_index = np.where(step_values == step)[0][0]
            step_dir = join(out_path, "{0:08d}".format(step))
            if not exists(step_dir):
                os.mkdir(step_dir)
            x_initial = lorenz_output["lorenz_x"][step_index].values
            y_initial = lorenz_output["lorenz_y"][step_index].values
            u_initial = y_initial.reshape(8, 32).sum(axis=1)
            launch_forecast_step(members, np.copy(x_initial), u_initial, f, u_model_path, random_updater_path,
                                 num_steps, num_random, time_step, random_seeds, step, x_only,
                                 call_param_once, out_path)
    else:

        pool = Pool(args.proc, maxtasksperchild=10)
        for step in initial_steps:
            step_index = np.where(step_values == step)[0][0]
            step_dir = join(out_path, "{0:08d}".format(step))
            if not exists(step_dir):
                os.mkdir(step_dir)
            x_initial = lorenz_output["lorenz_x"][step_index].values
            y_initial = lorenz_output["lorenz_y"][step_index].values
            u_initial = y_initial.reshape(8, 32).sum(axis=1)
            pool.apply_async(launch_forecast_step, (members, x_initial, u_initial, f, u_model_path,
                                                    random_updater_path, num_steps, num_random, time_step,
                                                    random_seeds, step, x_only, call_param_once, out_path,
                                                    num_tf_threads))
        pool.close()
        pool.join()
    return


def launch_forecast_step(members, x_initial, u_initial, f, u_model_path, random_updater_path, num_steps,
                         num_random, time_step, random_seeds, initial_step_value, x_only, call_param_once, out_path,
                         num_tf_threads):
    with open(random_updater_path, "rb") as random_updater_file:
        random_updater = pickle.load(random_updater_file)
    if u_model_path[-2:] == "h5":
        sess = K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=num_tf_threads,
                                                    inter_op_parallelism_threads=1))
        K.set_session(sess)
        u_model = SubModelGAN(u_model_path)
    else:
        with open(u_model_path, "rb") as u_model_file:
            u_model = pickle.load(u_model_file)
        sess = None
    for member in members:
        launch_forecast_member(member, x_initial, u_initial, f, u_model_path, u_model, random_updater, num_steps,
                               num_random, time_step, random_seeds[member], initial_step_value, x_only, call_param_once,
                               out_path)
    if sess is not None:
        sess.close()


def launch_forecast_member(member_number, x_initial, u_initial, f, u_model_path, u_model, random_updater, num_steps,
                           num_random, time_step, random_seed, initial_step_value, x_only, call_param_once, out_path):
    """
    Run a single Lorenz 96 forecast model with a specified x and u initial conditions and forcing. The output
    of the run is saved to a netCDF file.

    Args:
        member_number:
        x_initial:
        u_initial:
        f:
        u_model:
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
        np.random.seed(random_seed)
        if u_model_path == "h5":
            K.tf.set_random_seed(random_seed)
        print("Starting member {0:d}".format(member_number))
        forecast_out = run_lorenz96_forecast(x_initial, u_initial, f, u_model, random_updater, num_steps, num_random,
                                             time_step, x_only=x_only, call_param_once=call_param_once)
        forecast_out.attrs["initial_step"] = initial_step_value
        forecast_out.attrs["member"] = member_number
        forecast_out.attrs["u_model_path"] = u_model_path
        forecast_out.to_netcdf(join(out_path, "{0:08d}/lorenz_forecast_{0:08d}_{1:02d}.nc".format(initial_step_value,
                                                                                                  member_number)),
                               mode="w", encoding={"x": {"dtype": "float32", "zlib": True, "complevel": 2},
                                                   "u": {"dtype": "float32", "zlib": True, "complevel": 2}})
        forecast_out.close()
    except Exception as e:
        print(traceback.format_exc())
        raise e



if __name__ == "__main__":
    main()
