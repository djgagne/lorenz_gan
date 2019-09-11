#!/bin/bash
#PBS -N c_gan_70101
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
export PATH="/glade/u/home/dgagne/miniconda3/envs/ml/bin:$PATH"
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/exp_20_stoch/climate_gan_n_70101_c_dense.yaml -p 1 &> gan_70101_climate.log
