# Lorenz '96 with GAN parameterization

Reqiured Python Libraries:
* numpy
* scipy
* matplotlib
* xarray
* pandas
* jupyter
* keras
* tensorflow
* scikit-learn
* netcdf4
* pyyaml
* numba

## Installation
The Anaconda or Miniconda Python distribution is recommended for managing the dependencies for the 
lorenz_gan library. Install the library by searching for miniconda and downloading the appropriate
install script for your OS. Python 3.5 or later is recommended.

To install the dependencies required for this package, use the following command:
```bash
>>> conda install numpy scipy matplotlib xarray pandas ipython jupyter keras tensorflow scikit-learn netcdf4 pyyaml
```
Once the dependencies are installed, install the lorenz_gan package in your Python environment with
the following commands
```bash
>>> cd ~/lorenz_gan
>>> python setup.py install
```

If you are actively developing the code and do not want to keep re-installing the library after change,
you can use the following command to install a soft-link between the Python environment and the code:
```
>>> python setup.py develop
```

## Running the Lorenz '96 Model

The Lorenz '96 model can be run in both "truth" mode with a numerically resolved subgrid layer, and in "forecast" mode
with a parameterized subgrid forcing term. 

### train_lorenz_gan.py

The program ```train_lorenz_gan.py``` runs the Lorenz truth model and fits all of the parameterizations, including
the GAN. To run the program
```
>>> cd ~/lorenz_gan/
>>> python train_lorenz_gan.py config/lorenz.yaml -g
```

This option will run the Lorenz truth model and train a random number updater, histogram parameterization, 
polynomial regression parameterization, and a GAN parameterization. If you wish to re-train the parameterizations
without re-running the truth model, then you can add a ```-r``` argument after the config file. The ```-g``` option
runs the GAN training. If you do not wish to train the GAN, which can take a few minutes to run, then leave it off.

### run_lorenz_forecast.py

The program ```run_lorenz_forecast.py``` runs an ensemble of Lorenz forecast models with the parameterization specified
in the config file. 
```
>>> python run_lorenz_forecast.py config/forecast_poly.yaml -p 3
```
The ```-p``` argument specifies the number of processors to be used for generating forecasts. Each forecast member in
the ensemble is run in a separate process.