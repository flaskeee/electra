#!/bin/sh

#SBATCH --job-name=electra_finetuning
# #SBATCH --partition=windfall
#SBATCH --partition=standard
#SBATCH --account=nlp

#SBATCH --ntasks=4
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tf1
PY=`which python`

# look_back=$1
# look_forward=$2

DATA_DIR=pretraining_data

python3 compute_ngram.py sim --in_path pretrain_data/owt.tfrecords/ --out_path=pretraining_data/ngrams/owt.jaccard_5_5.pkl --vocab_size=`wc -l < pretraining_data/vocab.txt` --look_back=5 --look_forward=5 --count_file=./pretraining_data/ngrams/owt.context_5_5.pkl --sim_metric=jaccard
