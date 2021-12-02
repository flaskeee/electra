import fire
import sys
from pandas import DataFrame as df
import pandas as pd
from pathlib import Path
import regex as re
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Iterable, Tuple, Any
from itertools import chain


'''
class Table:
    """
    incomplete
    """

    def __init__(
            records: Tuple[Any, Any, Any],
            default_val_constructor=str,
    ):
        """
        records: tuples of (row, col, val)
        """
        name_to_idx = lambda xs: {
                x: i
                for i, x in enumerate(sorted(set(xs)))
            }

        records = list(records)
        row_name_to_idx = name_to_idx(r[0] for r in records)
        col_name_to_idx = name_to_idx(r[1] for r in records)

        table = [
                    [
                        default_val_constructor()
                        for _ in range(len(col_name_to_idx))
                        ]
                    for _ in range(len(row_name_to_idx))
                ]

        for r, c, v in records:
            table[row_name_to_idx[r]][col_name_to_idx[c]] = v

        self.row_name = list(row_name_to_idx.keys())
        self.col_name = list(col_name_to_idx.keys())
        self._table = table

    def to_csv():
        out = ',' + ''.join(self.col_name_to_idx) + '\n'
'''



def remove_common_options_fn(model_rec):
    all_opt_val = defaultdict(list)
    for mc in model_rec:
        new_model_dict = {}
        for o in mc['model'].split('.'):
            o = o.split('=')
            if len(o) == 1:
                k = o[0]
                v = None
            elif len(o) == 2:
                k, v = o
            else:
                raise RuntimeError('cannot handle option: ', o)
            all_opt_val[k].append(v)
            new_model_dict[k] = v
        mc['model'] = new_model_dict
    printed_opt = {
            k for k, v in all_opt_val.items()
            if not (len(v) > 1 and len(set(v)) == 1)
        }
    for mc in model_rec:
        mc['model'] = '.'.join(
                k if v is None else f'{k}={v}'
                for k, v in mc['model'].items()
                if k in printed_opt
            )


def main(
        experiment_dir='pretraining_data/models/',
        pattern='*',
        remove_common_options=False,

):
    records = []
    all_model_path = chain.from_iterable(
            Path(experiment_dir).glob(ptn)
            for ptn in pattern.split('|')
        )
    for model_path in all_model_path:
        model_name = model_path.name

        model_rec = {'model': model_name}

        for result_f_path in (model_path/'results').glob('*.txt'):
            m = re.match(
                    '(?P<task>\w+):[^0-9]+(?P<score>[0-9.]+).+',
                    open(result_f_path).read(),
                 )
            if m is not None:
                task, score = m.groups(('task', 'score'))
                model_rec[task] = score
        records.append(model_rec)

    if remove_common_options:
        remove_common_options_fn(records)

    pd.set_option('display.max_colwidth', None)
    table = df.from_records(records, index='model')
    table = table.sort_index(axis=0).sort_index(axis=1)
    print(table)


if __name__ == '__main__':
    fire.Fire(main)
