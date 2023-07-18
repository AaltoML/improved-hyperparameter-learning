#!/bin/sh
#SBATCH --time=48:00:00 
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --constraint="csl|skl"
#SBATCH --array=1-3

mkdir -p result

matlab -r "mcmc_student_t($SLURM_ARRAY_TASK_ID)"
