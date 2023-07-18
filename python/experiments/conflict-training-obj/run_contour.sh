#!/bin/sh
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --mem=1G
#SBATCH --array=2

mkdir -p experiment_results

python contour.py  --dataset_name ionosphere --sigma_grid_min -1 --sigma_grid_max 5 --l_grid_min -1 --l_grid_max 5 --model_name_id $SLURM_ARRAY_TASK_ID --grid_num 21 --k_fold_id 0