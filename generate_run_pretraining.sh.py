import json
import fire



def generate_run_name(d: dict):
    out = []
    for k, v in d.items():
        if type(v) in [int, float, str, bool]:
            out.append(
                k + '=' + str(v)
            )
    return '.'.join(out)


def get_ngram_options(n, cython_generator):
    if n < 0:
        return {}
    else:
        return {
            'ngram_generator': n,
            'ngram_pkl_path': f'owt_{n}_gram.pkl',
            'cython_generator': True
        }

def get_pretrain_data_options(pretrain_data):
    if not pretrain_data:
        return {}
    else:
        return {
                'pretrain_tfrecords': \
                        {
                            'mimic_iii': 'pretrain_data/mimic_iii.tfrecords',
                            'owt': 'pretrain_data/owt.tfrecords',
                        }[pretrain_data]
               }


def generate_script(options: dict, run_name: str):
    return f"""#!/bin/sh

#SBATCH --job-name=electra_pretraining
#SBATCH --partition=standard
#SBATCH --account=nlp

#SBATCH --ntasks=4
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tf1
PY=`which python`

DATA_DIR=/xdisk/bethard/jiachengz/electra_pretraining/pretrain_data

singularity exec --nv /groups/bethard/image/tensorflow_1_15.sif $PY run_pretraining.py --data-dir $DATA_DIR \\
    --model-name {run_name} \\
    --hparams '{json.dumps(options)}'
    """


def main(
        ngram=-1,
        cython_generator=True,
        pretrain_data='owt',
):
    run_name = generate_run_name(locals())
    options = {}
    options.update(
        get_ngram_options(ngram, cython_generator)
    )
    options.update(
        get_pretrain_data_options(pretrain_data)
    )

    return generate_script(options, run_name)


if __name__ == '__main__':
    (fire.Fire(main))

