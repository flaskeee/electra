#!/bin/sh
#SBATCH --job-name=build_openwebtxt_pretraining_dataset
#SBATCH --partition=windfall
#SBATCH --ntasks=12
#SBATCH --time=100:00:00
#SBATCH --nodes=1

eval "$(conda shell.bash hook)"
conda activate tf1

python3 build_openwebtext_pretraining_dataset.py --data-dir pretraining_data/ --num-processes 24
