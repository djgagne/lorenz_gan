#!/bin/bash
#PBS -N lorenz_climate
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=11:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate ml
cd /glade/u/home/dgagne/lorenz_gan
python -u run_lorenz_forecast.py config/climate_poly_chey.yaml -p 2 >& poly_clim.log &
python -u run_lorenz_forecast.py config/climate_gan_u_chey.yaml -p 2 >& gan_clim.log
