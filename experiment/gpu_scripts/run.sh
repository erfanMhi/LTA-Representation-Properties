#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miahi@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=30:00:00

cd $SLURM_SUBMIT_DIR/../../
module load python/3.6
source $HOME/rep_env/bin/activate
parallel --joblog experiment/gpu_scripts/log/'task_'"$SLURM_ARRAY_TASK_ID"'.log' < 'experiment/gpu_scripts/tasks_'"$SLURM_ARRAY_TASK_ID"'.txt'
