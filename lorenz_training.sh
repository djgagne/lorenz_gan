#!/bin/bash
#PBS -N lorenz_train
#PBS -A P54048000
#PBS -q regular
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -j oe
module unload ncarenv
source /glade/u/home/dgagne/.bash_profile
source activate deep
cd /glade/u/home/dgagne/lorenz_gan
#python -u train_lorenz_gan.py config/lorenz_train_f_20_n_000_c_dense.yaml -g &> lorenz_train.log
declare -a configs=("100" "101" "102" "103" "202" "203" "300" "301" "302" "303" "402" "403")
for config in ${configs[@]}; do
    echo $config
    python -u train_lorenz_gan.py config/exp_20_stoch/lorenz_train_f_20_n_${config}_c_dense.yaml -r -g &>> lorenz_train.log
done
