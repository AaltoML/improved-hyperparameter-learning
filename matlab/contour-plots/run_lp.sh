#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --partition=batch
#SBATCH --mem=500M
#SBATCH --array=0-440

mkdir -p lp/ionosphere

srun matlab -nojvm -nosplash -batch "MCMC_lp($SLURM_ARRAY_TASK_ID, 'ionosphere', -1, 5)"
