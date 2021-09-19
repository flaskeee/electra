import json
import fire


def generate_run_name(d: dict):
    out = []
    for k, v in d.items():
        if v in [int, float, str]:
            out.append(
                'k' + '=' + str(v)
            )
    return '.'.join(out)


def get_ngram_options(n):
    if n < 0:
        return {}
    else:
        return {
            'ngram_generator': n,
            'ngram_pkl_path': f'owt_{n}_gram.pkl'
        }


def generate_script(options: dict):
    return f"""
#!/bin/sh

#SBATCH --job-name=electra_pretraining
#SBATCH --partition=standard
#SBATCH --account=nlp

#SBATCH --ntasks=2
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tf1
PY=`which python`

DATA_DIR=/xdisk/bethard/jiachengz/electra_pretraining/openwebtext

singularity exec --nv /groups/bethard/image/tensorflow_1_15.sif python3 run_pretraining.py --data-dir $DATA_DIR \\
    --model-name {generate_run_name(options)} \\
    --hparams '{json.dumps(options)}'
    """


def main(
        ngram=-1,
):
    options = {}
    options.update(
        get_ngram_options(ngram)
    )

    return generate_script(options)


if __name__ == '__main__':
    print(fire.Fire(main))

