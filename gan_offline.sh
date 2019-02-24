#!/bin/bash
#PBS -N off_gan
#PBS -A NAML0001
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
echo `hostname`
source /glade/u/home/dgagne/.bashrc
export PATH="/glade/u/home/dgagne/miniconda3/envs/ml/bin:$PATH"
cd /glade/u/home/dgagne/lorenz_gan
python -u gan_offline_analysis.py config/exp_20_stoch/gan_offline_config.yaml -n 4 &> offline_gan.log
