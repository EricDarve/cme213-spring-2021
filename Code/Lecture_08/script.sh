#!/bin/bash
#SBATCH -o job_%j.out
#SBATCH -p CME
#SBATCH --gres=gpu:1

ls
pwd
./deviceQuery

while true; do
	ls
done
