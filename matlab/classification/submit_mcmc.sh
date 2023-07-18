#!/bin/sh
sbatch --output=print_output/job-42_%a.out run_mcmc_classification.sh 42

for index in {1..9}
do
    sbatch --output=print_output/job-${index}_%a.out run_mcmc_classification.sh $index
done
