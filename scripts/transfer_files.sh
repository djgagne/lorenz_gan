#!/bin/bash
#PBS -N l_trans
#PBS -A P54048000
#PBS -q regular
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
cd /glade/work/dgagne/
/glade/u/home/dgagne/rclone -v sync exp_20_stoch gd:Lorenz_GAN_docs/exp_20_stoch >& transfer.log
