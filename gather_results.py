import sys
from pandas import DataFrame as df
from pathlib import Path
import regex as re
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Iterable, Tuple, Any


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







records = []

for model_path in sys.stdin:
    model_path = Path(model_path.strip())
    model_name = model_path.name

    model_rec = {'model': model_name}

    for result_f_path in (model_path/'results').glob('*.txt'):
        task, score = re.match(
                '(?P<task>\w+):[^0-9]+(?P<score>[0-9.]+).+',
                open(result_f_path).read(),
             ).groups(('task', 'score'))
        model_rec[task] = score
    records.append(model_rec)


table = df.from_records(records, index='model')
table = table.sort_index(axis=0).sort_index(axis=1)
print(table)

