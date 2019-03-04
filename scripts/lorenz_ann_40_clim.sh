#!/bin/bash
#PBS -N l_clim_ann_4
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate ml
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/climate_ann_d_40_c.yaml -p 2 >& ann_40_clim.log
