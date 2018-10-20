#!/bin/bash
#PBS -N gan_w_20
#PBS -A P54048000
#PBS -q regular
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=36:ompthreads=36:mem=109GB
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate deep
cd /glade/u/home/dgagne/lorenz_gan
#python -u run_lorenz_forecast.py config/forecast_poly_chey.yaml -p 36
python -u run_lorenz_forecast.py config/forecast_20_gan_d_20_c_dense.yaml -p 36 >& weather_gan_20.log
