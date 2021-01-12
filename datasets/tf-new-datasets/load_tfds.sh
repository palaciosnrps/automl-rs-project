#!/bin/bash
#SBATCH --partition=graceGPU #Name of the GPUs partition
#SBATCH -x ethnode[22,23,33,34]
#SBATCH --mem=3G
#SBATCH -o /scratch/palacios/tests/log_dt.out #Standard output from the program
#SBATCH -e /scratch/palacios/tests/log_dt.err #Standard error from the program
#SBATCH -J load_tfds #Name of your program
#SBATCH --array 0-2 #create 3 jobs
source activate amluc
CUDA_VISIBLE_DEVICES=0
params=(\
'coffee_scenes.py' \
'savana_scenes.py' \
'brazildam_sentinel.py'\
)
srun tfds build --data_dir='/scratch/palacios/data' ${params[$SLURM_ARRAY_TASK_ID]}
