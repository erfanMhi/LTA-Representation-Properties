#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miahi@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00


module load python/3.7
source $HOME/lta-rep/bin/activate
python find_best.py