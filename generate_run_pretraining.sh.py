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


def get_sampler_options(ngram_generator, cython_generator, progressive_ngram, wrong_ngram, sim_generator, smoothing, sim_metric, sim_alpha, sim_progressive_alpha):
    assert not (ngram_generator > -1 and sim_generator > -1)
    if ngram_generator > -1:
        return {
            'ngram_generator': n,
            'word_count_pkl_path': f'pretraining_data/ngram/owt.{n}_gram.pkl',
            'cython_generator': True,
            'progressive_ngram': progressive_ngram,
            'wrong_ngram': wrong_ngram
        }
    elif sim_generator > -1:
        return {
            'sim_generator': True,
            'sim_alpha': sim_alpha,
            'sim_progressive_alpha': sim_progressive_alpha,
            'word_count_pkl_path': f"pretraining_data/ngrams/owt.{sim_metric}_{sim_generator}_{sim_generator}.{'smoothing.' if smoothing else ''}pkl",
        }
    else:
        return {}

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
        sim_generator=-1,
        sim_metric='',
        sim_alpha=1,
        sim_progressive_alpha=False,
        progressive_ngram=False,
        cython_generator=True,
        pretrain_data='owt',
        wrong_ngram=False,
        ngram_mod='none',
        smoothing=False,
        debug=False,
):
    cmd_options = dict(locals())
    if cmd_options.pop('debug'):
        run_name = 'debug'
    else:
        run_name = generate_run_name(cmd_options)
    options = {}
    options.update(
        get_sampler_options(ngram, cython_generator, progressive_ngram, wrong_ngram, sim_generator, smoothing, sim_metric, sim_alpha, sim_progressive_alpha)
    )
    options.update(
        get_pretrain_data_options(pretrain_data)
    )
    options['ngram_mod'] = ngram_mod

    return generate_script(options, run_name)


if __name__ == '__main__':
    (fire.Fire(main))

