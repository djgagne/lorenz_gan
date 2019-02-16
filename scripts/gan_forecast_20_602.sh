#!/bin/bash
#PBS -N f_gan_602
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
export PATH="/glade/u/home/dgagne/miniconda3/envs/ml/bin:$PATH"
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/exp_20_stoch/forecast_20_gan_n_602_c_dense.yaml -p 36 &> gan_602_forecast_20.log
