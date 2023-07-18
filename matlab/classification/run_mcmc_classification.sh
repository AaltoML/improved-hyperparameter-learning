#!/bin/sh
#SBATCH --time=110:00:00 
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --constraint="csl|skl"
#SBATCH --array=0-26

mkdir -p result

matlab -r "mcmc_classification($SLURM_ARRAY_TASK_ID, $1)"
