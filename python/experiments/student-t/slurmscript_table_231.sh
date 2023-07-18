#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --mem=5G
#SBATCH --array=0-4

mkdir -p student_t_results

python student_t.py --model_name cvi --dataset_name neal --total_step 2000 --k_fold_id $SLURM_ARRAY_TASK_ID