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


def get_ngram_options(n, cython_generator, progressive_ngram, wrong_ngram):
    if n < 0:
        return {}
    else:
        return {
            'ngram_generator': n,
            'ngram_pkl_path': f'owt_{n}_gram.pkl',
            'cython_generator': True,
            'progressive_ngram': progressive_ngram,
            'wrong_ngram': wrong_ngram
        }

def get_pretrain_data_options(pretrain_data):
    if not pretrain_data:
        return {}
    else:
        return {
                'pretrain_tfrecords': \
                        {
                            'mimic_iii': 'pretraining_data/mimic_iii.tfrecords',
                            'owt': 'pretraining_data/owt.tfrecords',
                        }[pretrain_data] + '/pretrain_data.tfrecord*'
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

rm -r $DATA_DIR/models/debug

singularity exec --nv /groups/bethard/image/tensorflow_1_15.sif $PY run_pretraining.py --data-dir $DATA_DIR \\
    --model-name {run_name} \\
    --hparams '{json.dumps(options)}'
    """


def main(
        ngram=-1,
        progressive_ngram=False,
        cython_generator=True,
        pretrain_data='owt',
        wrong_ngram=False,
        ngram_mod='none',
        debug=False,
):
    cmd_options = dict(locals())
    if cmd_options.pop('debug'):
        run_name = 'debug'
    else:
        run_name = generate_run_name(cmd_options)
    options = {}
    options.update(
        get_ngram_options(ngram, cython_generator, progressive_ngram, wrong_ngram)
    )
    options.update(
        get_pretrain_data_options(pretrain_data)
    )
    options['ngram_mod'] = ngram_mod

    return generate_script(options, run_name)


if __name__ == '__main__':
    (fire.Fire(main))

