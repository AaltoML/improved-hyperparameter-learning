#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=3
#SBATCH --partition=batch
#SBATCH --mem=1G
#SBATCH --array=0-440

mkdir -p lml/ionosphere


srun matlab -nojvm -nosplash -batch "MCMC_lml($SLURM_ARRAY_TASK_ID, 'ionosphere', -1, 5)"


