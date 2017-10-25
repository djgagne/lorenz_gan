import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd


@jit(nopython=True, cache=True)
def l96_truth_step(X, Y, h, F, b, c):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Y[k * J: (k + 1) * J])
    for j in range(J * K):
        dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1]) - c * Y[j] + h * c / b * X[int(j / J)]
    return dXdt, dYdt


@jit(nopython=True, cache=True)
def run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Args:
        X (1D ndarray): Initial X values.
        Y (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        F (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.

    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    X_out = np.zeros((num_steps + 1, X.size))
    Y_out = np.zeros((num_steps + 1, Y.size))
    steps = np.arange(num_steps + 1)
    times = steps * time_step
    X_out[0] = X
    Y_out[0] = Y
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)
    k1_dYdt = np.zeros(Y.shape)
    k2_dYdt = np.zeros(Y.shape)
    k3_dYdt = np.zeros(Y.shape)
    k4_dYdt = np.zeros(Y.shape)
    for n in range(1, num_steps + 1):
        print(n)
        k1_dXdt[:], k1_dYdt[:] = l96_truth_step(X, Y, h, F, b, c)
        k2_dXdt[:], k2_dYdt[:] = l96_truth_step(X + k1_dXdt * time_step / 2,
                                          Y + k1_dYdt * time_step / 2,
                                          h, F, b, c)
        k3_dXdt[:], k3_dYdt[:] = l96_truth_step(X + k2_dXdt * time_step / 2,
                                          Y + k2_dYdt * time_step / 2,
                                          h, F, b, c)
        k4_dXdt[:], k4_dYdt[:] = l96_truth_step(X + k3_dXdt * time_step,
                                          Y + k3_dYdt * time_step,
                                          h, F, b, c)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * time_step
        Y += (k1_dYdt + 2 * k2_dYdt + 2 * k3_dYdt + k4_dYdt) / 6 * time_step
        X_out[n] = X
        Y_out[n] = Y
    return X_out, Y_out, times, steps


@jit(nopython=True, cache=True)
def l96_forecast_step(X, F):
    K = X.size
    dXdt = np.zeros(X.size)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F
    return dXdt


def run_lorenz96_forecast(X, F, u_model, time_step, num_steps, random_seed):
    np.random.seed(random_seed)
    X_out = np.zeros((num_steps + 1, X.size))
    steps = np.arange(num_steps + 1)
    times = steps * time_step
    X_out[0] = X
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    for n in range(1, num_steps + 1):
        k1_dXdt[:] = l96_forecast_step(X, F)
        k2_dXdt[:] = l96_forecast_step(X + k1_dXdt * time_step, F)
        X += 0.5 * (k1_dXdt + k2_dXdt) * time_step
        X_out[n] = X
    return X_out, times, steps

def process_lorenz_data(X_out, Y_out, times, steps, cond_inputs, J, x_skip, t_skip):
    """
    Sample from Lorenz model output and reformat the data into a format more amenable to machine learning.


    Args:
        X_out (ndarray): Lorenz 96 model output
        Y_out (ndarray):
        cond_inputs (int): number of lagged X values to associate with each set of Ys
        J (int): number of Y variables per X variable
        x_skip (int): number of X variables to skip when sampling the data
        t_skip (int): number of time steps to skip when sampling the data

    Returns:
        combined_data: pandas DataFrame
    """
    X_series_list = []
    Y_series_list = []
    x_s = np.arange(0, X_out.shape[1], x_skip)
    t_s = np.arange(cond_inputs, X_out.shape[0], t_skip)
    time_list = []
    step_list = []
    x_list = []
    for k in x_s:
        X_series_list.append(np.stack([X_out[i:i-cond_inputs:-1, k]
                             for i in t_s], axis=0))
        Y_series_list.append(Y_out[cond_inputs::t_skip, k * J: (k+1) * J])
        time_list.append(times[cond_inputs::t_skip])
        step_list.append(steps[cond_inputs::t_skip])
        x_list.append(np.ones(time_list[-1].size) * k)
    X_series = np.vstack(X_series_list)
    Y_series = np.vstack(Y_series_list)
    x_cols = ["X_t"]
    if cond_inputs > 1:
        x_cols += ["X_t-{0:d}".format(t) for t in range(1, cond_inputs)]
    y_cols = ["Y_{0:d}".format(y) for y in range(J)]
    combined_data = pd.DataFrame(X_series, columns=x_cols)
    combined_data = pd.concat([combined_data, pd.DataFrame(Y_series, columns=y_cols)], axis=1)
    combined_data.loc[:, "time"] = np.concatenate(time_list)
    combined_data.loc[:, "step"] = np.concatenate(step_list)
    combined_data.loc[:, "x_index"] = np.concatenate(x_list)
    return combined_data[["x_index", "step", "time"] + x_cols + y_cols]


def save_lorenz_output(X_out, Y_out, times, steps, model_attrs, out_file):
    """
    Write Lorenz 96 truth model output to a netCDF file.

    Args:
        X_out (ndarray): X values from the model run
        Y_out (ndarray): Y values from the model run
        times (ndarray): time steps of model in units of MTU
        steps (ndarray): integer integration step values
        model_attrs (dict): dictionary of model attributes
        out_file: Name of the netCDF file

    Returns:

    """
    data_vars = dict()
    data_vars["time"] = xr.DataArray(times, dims=["time"], name="time", attrs={"long_name": "integration time",
                                                                               "units": "MTU"})
    data_vars["step"] = xr.DataArray(steps, dims=["time"], name="step", attrs={"long_name": "integration step",
                                                                               "units": ""})
    data_vars["lorenz_x"] = xr.DataArray(X_out, coords={"time": data_vars["time"], "x": np.arange(X_out.shape[1])},
                                         dims=["time", "x"], name="lorenz_X", attrs={"long_name": "lorenz_x",
                                                                                     "units": ""})
    data_vars["lorenz_y"] = xr.DataArray(Y_out, coords={"time": times, "y": np.arange(Y_out.shape[1])},
                                         dims=["time", "y"], name="lorenz_Y", attrs={"long_name": "lorenz_y",
                                                                                     "units": ""})
    l_ds = xr.Dataset(data_vars=data_vars, attrs=model_attrs)
    l_ds.to_netcdf(out_file, "w", encoding={"lorenz_x" : {"zlib": True, "complevel": 2},
                                            "lorenz_y": {"zlib": True, "complevel": 2}})
    return


def save_lorenz_series(X_series, Y_series):
    return

def main():
    K = 8
    J = 32
    X = np.zeros(K)
    Y = np.zeros(J * K)
    X[0] = 1
    Y[0] = 1
    h = 1
    b = 10.0
    c = 10.0
    time_step = 0.001
    num_steps = 50000
    F = 30.0
    X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps)
    print(X_out.max(), X_out.min())
    plt.figure(figsize=(8, 10))
    plt.pcolormesh(np.arange(K + 1), times, X_out, cmap="RdBu_r")
    plt.title("Lorenz '96 X Truth")
    plt.colorbar()

    plt.figure(figsize=(8, 10))
    plt.pcolormesh(np.arange(J * K + 1), times, Y_out, cmap="RdBu_r")
    plt.xticks(np.arange(0, J * K, J))
    plt.title("Lorenz '96 Y Truth")
    plt.colorbar()

    plt.figure(figsize=(10, 5))
    plt.plot(times, X_out[:, 0], label="X (0)")
    plt.plot(times, X_out[:, 3], label="X (3)")
    plt.legend(loc=0)
    plt.xlabel("Time (MTU)")
    plt.ylabel("X Values")
    plt.show()
    return 

        
if __name__ == "__main__":
    main()

