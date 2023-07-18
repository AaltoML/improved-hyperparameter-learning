#!/bin/sh
for index in {1..9}
do
    sbatch run_la.sh $index
    sbatch run_ep.sh $index
    sbatch run_vi.sh $index
    sbatch run_epcvi.sh $index
done

sbatch run_la.sh 42
sbatch run_ep.sh 42
sbatch run_vi.sh 42
sbatch run_epcvi.sh 42
