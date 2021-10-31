from itertools import chain
from pathlib import Path

from tqdm import tqdm
import tensorflow as tf
from typing import Union, Iterable, List

tf.enable_eager_execution()


def parse_tfrecords_for_ids(path: Union[str, Path], progress_bar=False) -> Iterable[List[int]]:
  """
  Returns an iterable of List of ids
  """
  if isinstance(path, str):
    path = Path(path)

  records = list(path.iterdir()) if path.is_dir() else [path]
  if progress_bar:
    records = tqdm(records)

  return chain.from_iterable(
    (
      tf.train.Example.FromString(e).features.feature['input_ids'].int64_list.value
      for e in tf.python_io.tf_record_iterator(str(r_path))  # tf.data.TFRecordDataset(str(r_path))
    )
    for r_path in records
  )

