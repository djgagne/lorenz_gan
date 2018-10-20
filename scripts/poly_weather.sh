#!/bin/bash
#PBS -N poly_w
#PBS -A P54048000
#PBS -q regular
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate deep
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/forecast_poly_add_n_c_dense.yaml -p 36 >& weather_poly_add.log
