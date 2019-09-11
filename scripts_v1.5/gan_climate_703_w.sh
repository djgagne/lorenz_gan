#!/bin/bash
#PBS -N c_gan_703
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
export PATH="/glade/u/home/dgagne/miniconda3/envs/ml/bin:$PATH"
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/exp_20_stoch/climate_gan_n_703_c_dense_w.yaml -p 1 &> gan_703_climate.log
