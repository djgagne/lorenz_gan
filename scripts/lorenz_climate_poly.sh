#!/bin/bash
#PBS -N l_clim_poly
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=3:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate ml
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/climate_poly_add_n_c_dense.yaml -p 2 >& poly_add_clim.log 
