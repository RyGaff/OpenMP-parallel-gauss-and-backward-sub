#!/bin/bash
#
#SBATCH --job-name=par_gRow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=results_Back_Row.txt

for size in 2500 5000 7500 10000 15000 20000; do
    echo $size x $size
    echo SERIAL:
    ./gauss -t $size
    echo SERIAL PARALLEL VERSION:
    ./par_gauss_serial -t $size 

    echo PARALLEL:
    for t in 1 2 4 8 16 32; do
        OMP_NUM_THREADS=$t ./par_gauss -t $size
    done
    echo
done

