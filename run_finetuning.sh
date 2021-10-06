#!/bin/sh

#SBATCH --job-name=electra_finetuning
# #SBATCH --partition=windfall
#SBATCH --partition=standard
#SBATCH --account=nlp

#SBATCH --ntasks=2
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tf1
PY=`which python`

run_name=$1
task_name=$2
model_name=$3

DATA_DIR=/xdisk/bethard/jiachengz/electra_pretraining/pretrain_data
hparam="{\"task_names\": [\"$task_name\"]}"

singularity exec --nv /groups/bethard/image/tensorflow_1_15.sif $PY run_finetuning.py --data-dir $DATA_DIR --model-name $model_name --hparams "$hparam"
