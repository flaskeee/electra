from tqdm import tqdm
import pickle
from pathlib import Path
from itertools import chain, tee
from collections import Counter
from typing import Iterable, Union, List
import numpy as np
from scipy.sparse import dok_matrix
from multiprocessing import pool
from functools import partial

import tensorflow as tf
tf.enable_eager_execution()


'''
path = '/xdisk/bethard/jiachengz/electra_pretraining/openwebtext/pretrain_tfrecords/pretrain_data.tfrecord-9-of-1000'
es = [e for e in tf.python_io.tf_record_iterator(path)]
es = [
    tf.train.Example.FromString(e)
    tf.data.TFRecordDataset(path)
]
e_ids: List[int] = tf.train.Example.FromString(es[0]).features.feature['input_ids'].int64_list.value
'''


def parse_tfrecords_for_ids(path: Union[str, Path]) -> Iterable[List[int]]:
    """
    Returns an iterable of List of ids
    """
    if isinstance(path, str):
        path = Path(path)

    records = list(path.iterdir()) if path.is_dir() else [path]
    return chain.from_iterable(
        (
            tf.train.Example.FromString(e).features.feature['input_ids'].int64_list.value
            for e in tf.python_io.tf_record_iterator(str(r_path))  # tf.data.TFRecordDataset(str(r_path))
        )
        for r_path in tqdm(records)
    )


def parallelize_ngram(fn, aggregate_fn=sum):
    def new_fn(samples, vocab_size):
        with pool.Pool() as p:
            return aggregate_fn(
                p.map(
                    partial(fn, vocab_size=vocab_size),
                    samples
                )
            )
    return new_fn


@parallelize_ngram
def mono_gram(
    samples: Iterable[Iterable[int]],
    vocab_size: int,
):
    counter = np.zeros(vocab_size)
    for v in chain.from_iterable(samples):
        counter[v] += 1
    return counter


@parallelize_ngram
def bi_gram(
    samples: Iterable[Iterable[int]],
    vocab_size: int,
):
    counter = np.zeros((vocab_size, vocab_size))
    for s in samples:
        prev_iter, curr_iter = tee(s, 2)
        next(curr_iter, None)
        for prev, curr in zip(prev_iter, curr_iter):
            counter[prev, curr] += 1

    return counter


def pickle_output(fn, out_path):
    def wrapped(*args, **kwargs):
        pickle.dump(
            fn(*args, **kwargs),
            open(out_path, 'wb')
        )
    return wrapped


class Launcher:
    def __init__(self, in_path, vocab_size: int, out_path='out.pkl',):
        self.sample_generator = parse_tfrecords_for_ids(in_path)
        self.output_path = out_path
        self.vocab_size = int(vocab_size)

    def monogram(self,):
        pickle_output(mono_gram, out_path=self.output_path)(samples=self.sample_generator, vocab_size=self.vocab_size)

    def bigram(self,):
        pickle_output(bi_gram, out_path=self.output_path)(samples=self.sample_generator, vocab_size=self.vocab_size)


if __name__ == '__main__':
    import fire
    fire.Fire(Launcher)
