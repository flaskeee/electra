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

from cos_similarity import count_context
import tensorflow as tf
tf.enable_eager_execution()
import torch
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tf_utils import parse_tfrecords_for_ids


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


def sim(
    samples: Iterable[Iterable[int]],
    vocab_size: int,
    look_back: int,
    look_forward: int,
    count_file: str = None,
    smoothing=False,
    sim_metric='cos',
):
    if count_file is not None:
        counts = pickle.load(open(count_file, 'rb'))
    else:
        counts = count_context(samples, vocab_size, look_back, look_forward)
    counts_float = counts.astype(np.float64)

    if smoothing:
        for i in np.argsort(np.sum(counts_float, axis=0))[-20:]:
            counts_float[:, i] = 0
        
    if sim_metric == 'cos':
        norm = np.sqrt(
            np.sum(counts_float ** 2, axis=1, keepdims=True)
        ).astype(np.float32)
        norm_product = norm @ norm.T
        similarity = (counts_float @ counts_float.T) / norm_product
    elif sim_metric == 'jaccard':
        counts = torch.from_numpy(counts).to(device=torch_device)
        neighbor_count = (counts > 0).float()
        print('begining')
        intersection = neighbor_count @ neighbor_count.T
        print('done')
        unique_neighbors = torch.sum(neighbor_count, dim=1, keepdim=True)
        union = unique_neighbors + unique_neighbors.T - intersection
        similarity = intersection / union
        similarity = np.nan_to_num(similarity.cpu().numpy(), copy=False)

    np.fill_diagonal(similarity, 0)
    np.nan_to_num(similarity, copy=False)
    return similarity.astype(np.float32)


def pickle_output(fn, out_path):
    def wrapped(*args, **kwargs):
        pickle.dump(
            fn(*args, **kwargs),
            open(out_path, 'wb')
        )
    return wrapped


class Launcher:
    def __init__(self, in_path, vocab_size: int, out_path='out.pkl', look_back=1, look_forward=1):
        self.sample_generator = parse_tfrecords_for_ids(in_path, progress_bar=True)
        self.output_path = out_path
        self.vocab_size = int(vocab_size)
        self.look_back = look_back
        self.look_forward = look_forward
        self.count_file = count_file
        self.smoothing = smoothing
        self.sim_metric = sim_metric

    def monogram(self,):
        pickle_output(mono_gram, out_path=self.output_path)(samples=self.sample_generator, vocab_size=self.vocab_size)

    def bigram(self,):
        pickle_output(bi_gram, out_path=self.output_path)(samples=self.sample_generator, vocab_size=self.vocab_size)

    def sim(self,):
        pickle_output(sim, out_path=self.output_path)(
            samples=self.sample_generator,
            vocab_size=self.vocab_size,
            look_back=self.look_back,
            look_forward=self.look_forward,
            count_file=self.count_file,
            smoothing=self.smoothing,
            sim_metric=self.sim_metric,
        )

    def count_context(self,):
        pickle_output(count_context,  out_path=self.output_path)(
            samples=self.sample_generator,
            vocab_size=self.vocab_size,
            look_back=self.look_back,
            look_forward=self.look_forward,
        )

if __name__ == '__main__':
    import fire
    fire.Fire(Launcher)
