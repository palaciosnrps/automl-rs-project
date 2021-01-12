#!/bin/bash
#SBATCH --partition=graceGPU #Name of the GPUs partition
#SBATCH -x ethnode[22,23,33,34]
#SBATCH --mem=3G
#SBATCH -o /scratch/palacios/tests/log_newdt.out #Standard output from the program
#SBATCH -e /scratch/palacios/tests/log_newdt.err #Standard error from the program
#SBATCH -J new_tfds #Name of your program

source activate amluc
CUDA_VISIBLE_DEVICES=0
srun  /scratch/palacios/anaconda3/envs/amluc/bin/python /scratch/palacios/repos/automl-rs/datasets/datasets.py
