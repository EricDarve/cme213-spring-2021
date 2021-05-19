#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=CME
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=fp
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------
echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $SLURM_SUBMIT_DIR`"
echo
echo Output from code
echo ----------------
### end of information preamble

cd $SLURM_SUBMIT_DIR

make

mpirun gather_ring

# salloc -N 1 -n 4 --partition=CME mpirun gather_ring
