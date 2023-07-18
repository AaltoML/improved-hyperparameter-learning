#!/bin/sh
#SBATCH --time=5:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=3
#SBATCH --mem=2G

mkdir -p experiment_results

python trajectory.py  --dataset_name ionosphere --setting 1 --steps 2000