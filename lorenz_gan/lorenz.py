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
def run_lorenz96_truth(x_initial, y_initial, h, f, b, c, time_step, num_steps, burn_in, skip):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        f (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival

    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    archive_steps = (num_steps - burn_in) // skip
    print(archive_steps)
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            print(n, n * time_step)
        k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, f, b, c)
        k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                                y + k1_dydt * time_step / 2,
                                                h, f, b, c)
        k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                                y + k2_dydt * time_step / 2,
                                                h, f, b, c)
        k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                                y + k3_dydt * time_step,
                                                h, f, b, c)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = y
            i += 1
    return x_out, y_out, times, steps


@jit(nopython=True, cache=True)
def l96_forecast_step(X, F):
    """
    Calculate the tendency of the Lorenz 96 Forecast Model dynamics

    Args:
        X (ndarray): Array of x values at a given time step
        F (float): Forcing value

    Returns:
        dXdt: the time tendency of the Xs
    """
    K = X.size
    dXdt = np.zeros(X.size)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F
    return dXdt


def run_lorenz96_forecast(x_initial, u_initial, f, u_model, random_updater, num_steps,
                          num_random, time_step=0.005, x_only=True, call_param_once=False):
    """
    Integrate the Lorenz 96 forecast model forward in time from a specified initial state with
    a parameterized subgrid forcing model. The u_model should contain a predict method that
    returns an array of U values

    Args:
        x_initial (ndarray): Initial x values
        f (float): Constant forcing value
        u_model: Sub-grid parameterization model
        random_updater: Method for updating the random values
        num_steps (int): Number of integration time steps
        num_random (int): Number of random values used by the parameterization
        time_step (float): Size of the integration time step in MTUs
        x_only (bool): Pass only the x information into the parameterization if True
        call_param_once (bool): Call the parameterization once per integration time step if True.

    Returns:
        output: xarray Dataset containing X and U values
    """
    order = 4
    time_inc = np.array([0.5, 0.5, 1])
    steps = np.arange(num_steps)
    times = steps * time_step
    x_u_curr = np.zeros((2, x_initial.shape[0]))
    x_u_curr[0] = x_initial[:]
    x_u_curr[1] = u_initial[:]
    coords = {"step": steps, "x_size": np.arange(x_initial.size)}
    X_out = xr.DataArray(np.zeros((num_steps, x_initial.size)),
                         coords=coords, dims=("step", "x_size"),
                         attrs={"long_name": "x values"})
    U_out = xr.DataArray(np.zeros((num_steps, x_initial.size)),
                         coords=coords, dims=("step", "x_size"),
                         attrs={"long_name": "u values"})
    X_out[0] = x_initial
    U_out[0] = u_initial
    k_dXdt = np.zeros((order, x_initial.shape[0]))
    random_values = np.random.normal(size=(x_initial.size, num_random))
    for n in range(1, num_steps):
        if n % 10 == 0:
            print(n)
        for o in range(order):
            if o == 0:
                if x_only:
                    x_u_curr[1] = u_model.predict(x_u_curr[0:1].T, random_values)
                else:
                    x_u_curr[1] = u_model.predict(x_u_curr.T, random_values)
                U_out[n] = x_u_curr[1]
            elif o > 0 and not call_param_once:
                if x_only:
                    x_u_curr[1] = u_model.predict(x_u_curr[0:1].T, random_values)
                else:
                    x_u_curr[1] = u_model.predict(x_u_curr.T, random_values)
            k_dXdt[o] = l96_forecast_step(x_u_curr[0], f) - x_u_curr[1]
            if o < order - 1:
                x_u_curr[0] = X_out[n - 1] + k_dXdt[o] * time_inc[o] * time_step
        x_u_curr[0] = X_out[n - 1] + (k_dXdt[0] + 2 * k_dXdt[1] + 2 * k_dXdt[2] + k_dXdt[3]) / 6 * time_step
        X_out[n] = x_u_curr[0]
        random_values = random_updater.update(random_values)
    output = xr.Dataset(data_vars=dict(x=X_out, u=U_out, time=times),
                        coords=coords)
    return output


def process_lorenz_data(X_out, Y_out, times, steps, J, x_skip, t_skip, u_scale):
    """
    Sample from Lorenz model output and reformat the data into a format more amenable to machine learning.


    Args:
        X_out (ndarray): Lorenz 96 model output
        Y_out (ndarray):
        J (int): number of Y variables per X variable
        x_skip (int): number of X variables to skip when sampling the data
        t_skip (int): number of time steps to skip when sampling the data

    Returns:
        combined_data: pandas DataFrame
    """
    X_series_list = []
    Y_series_list = []
    U_series_list = []
    x_s = np.arange(0, X_out.shape[1], x_skip)
    t_s = np.arange(0, X_out.shape[0] - 1, t_skip)
    time_list = []
    step_list = []
    x_list = []
    for k in x_s:
        X_series_list.append(X_out[t_s, k: k + 1])
        Y_series_list.append(Y_out[1::t_skip, k * J: (k+1) * J])
        U_series_list.append(np.expand_dims(u_scale * Y_out[t_s, k * J: (k+1) * J].sum(axis=1), 1))
        time_list.append(times[t_s])
        step_list.append(steps[t_s])
        x_list.append(np.ones(time_list[-1].size) * k)

    X_series = np.vstack(X_series_list)
    Y_series = np.vstack(Y_series_list)
    U_series = np.vstack(U_series_list)
    x_cols = ["X_t"]
    y_cols = ["Y_{0:d}".format(y) for y in range(J)]
    u_col = "U_t"
    combined_data = pd.DataFrame(X_series, columns=x_cols)
    combined_data = pd.concat([combined_data, pd.DataFrame(Y_series, columns=y_cols)], axis=1)
    combined_data.loc[:, "time"] = np.concatenate(time_list)
    combined_data.loc[:, "step"] = np.concatenate(step_list)
    combined_data.loc[:, "x_index"] = np.concatenate(x_list)
    combined_data.loc[:, u_col] = U_series
    combined_data.loc[:, "u_scale"] = u_scale
    out_cols = ["x_index", "step", "time"] + x_cols + y_cols + [u_col, 'u_scale']
    return combined_data.loc[:, out_cols]


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

