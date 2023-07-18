#!/bin/sh
#SBATCH --time=40:00:00
#SBATCH --partition=batch
#SBATCH --mem=5G
#SBATCH --array=0-4

mkdir -p student_t_results

python student_t.py --model_name cvi --dataset_name stock --total_step 5000 --k_fold_id $SLURM_ARRAY_TASK_ID