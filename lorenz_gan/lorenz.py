import numpy as np
from numba import jit
import matplotlib.pyplot as plt


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
    return X_out, Y_out


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
    mtu = np.arange(0, num_steps * time_step + time_step, time_step)
    F = 30.0
    X_out, Y_out = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps)
    print(X_out.max(), X_out.min())
    plt.figure(figsize=(8, 10))
    plt.pcolormesh(np.arange(K + 1), mtu, X_out, cmap="RdBu_r")
    plt.title("Lorenz '96 X Truth")
    plt.colorbar()

    plt.figure(figsize=(8, 10))
    plt.pcolormesh(np.arange(J * K + 1), mtu, Y_out, cmap="RdBu_r")
    plt.xticks(np.arange(0, J * K, J))
    plt.title("Lorenz '96 Y Truth")
    plt.colorbar()

    plt.figure(figsize=(10, 5))
    plt.plot(mtu, X_out[:, 0], label="X (0)")
    plt.plot(mtu, X_out[:, 3], label="X (3)")
    plt.legend(loc=0)
    plt.xlabel("Time (MTU)")
    plt.ylabel("X Values")
    plt.show()
    return 

        
if __name__ == "__main__":
    main()

