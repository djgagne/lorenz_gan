#!/bin/bash
#PBS -N lorenz_train
#PBS -A P54048000
#PBS -q regular
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate deep
cd /glade/u/home/dgagne/lorenz_gan
python -u train_lorenz_gan.py config/lorenz_train_f_20_d_0_c.yaml -g
python -u train_lorenz_gan.py config/lorenz_train_f_20_d_20_c.yaml -r -g
python -u train_lorenz_gan.py config/lorenz_train_f_20_d_40_c.yaml -r -g
