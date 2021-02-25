#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00


chmod +x task*
cd $SLURM_SUBMIT_DIR/../../
module load python/3.6
source $HOME/torch1env/bin/activate
'experiment/scripts/tasks_'"$SLURM_ARRAY_TASK_ID"'.sh'
