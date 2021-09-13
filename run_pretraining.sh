#!/bin/sh

#SBATCH --job-name=electra_pretraining
#SBATCH --account=nlp
#SBATCH --partition=standard
#SBATCH --ntasks=2
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tf1


DATA_DIR=/xdisk/bethard/jiachengz/electra_pretraining/openwebtext
# hparam='{"debug": true}'
# hparam='{"uniform_generator": true, "num_train_steps": 108000}'
hparam='{"ngram_generator": 1, "ngram_pkl_path": "owt_monogram.pkl"}'

singularity exec --nv /groups/bethard/image/tensorflow_1_15.sif python3 run_pretraining.py --data-dir $DATA_DIR --model-name electra_small_owt_108000steps_monogramgenerator --hparams "$hparam"
