#! /bin/bash
#
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=20
#PBS -W group_list=newriver
#PBS -q dev_q
#PBS -j oe
#PBS -A CMDA3634SP18

cd $PBS_O_WORKDIR

module purge
module load gcc openmpi

mpicc -o main main.c functions.c -lm

mpiexec -np 1 ./main
mpiexec -np 2 ./main
mpiexec -np 4 ./main
mpiexec -np 8 ./main
mpiexec -np 12 ./main
mpiexec -np 16 ./main
mpiexec -np 20 ./main
