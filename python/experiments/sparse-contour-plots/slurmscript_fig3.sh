#!/bin/sh
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=3
#SBATCH --mem=3G
#SBATCH --array=0-3

mkdir -p experiment_results

python reduce_Z.py --dataset_name ionosphere --sigma_grid_min -1 --sigma_grid_max 5 --l_grid_min -1 --l_grid_max 5 --grid_num 21 --ratio_id $SLURM_ARRAY_TASK_ID

