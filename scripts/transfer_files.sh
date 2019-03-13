#!/bin/bash -l
#SBATCH --job-name=l_trans
#SBATCH --account=NAML0001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --partition=dav
#SBATCH --output=trans.out.%j
module unload ncarenv
source /glade/u/home/dgagne/.bashrc
cd /glade/work/dgagne/
/glade/u/home/dgagne/rclone -v sync exp_20_stoch/v2 gd:Lorenz_GAN_docs/exp_20_stoch/v2 >& transfer.log
