import atexit
import signal
import uuid
from itertools import chain
from pathlib import Path

from tqdm import tqdm
import tensorflow as tf
from typing import Union, Iterable, List



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


def ckpt_iter(ckpt_path) -> Path:
  """
  Work around to iterate through all historical checkpoints
    !!! yields the path to a temp dir that contains symlinks to the next ckpt files
    TF only reads the latest ckpt in a directory, hence we:
      - make a temp dir
      - make symlink to a certain checkpoint
      - yield path to this temp dir
      - clean up, move to the next
  """
  import regex as re
  ckpt_path = Path(ckpt_path)
  all_ckpt_files = list(ckpt_path.iterdir())
  ckpt_idx2stem = {}
  for f_name in all_ckpt_files:
    stem = f_name.stem
    if 'ckpt-' in stem:
      idx = int(re.findall(r'\d+', stem)[-1])
      ckpt_idx2stem[idx] = stem

  temp_dir = Path(
    'tmp.ckpt_iter.' + str(uuid.uuid4())
  )
  def clean_up():
    for symlink in temp_dir.iterdir():
      symlink.unlink()
    temp_dir.rmdir()
  atexit.register(clean_up())
  signal.signal(signal.SIGINT, clean_up)
  signal.signal(signal.SIGTERM, clean_up)
  for idx, stem in sorted(ckpt_idx2stem.items(), key=lambda x: x[0]):
    temp_dir.mkdir()
    for f_name in all_ckpt_files:
      if f_name.stem == stem:
        symlink = temp_dir / f_name.name
        symlink.symlink_to(f_name)
        symlink.chmod(400)
    yield temp_dir
    clean_up()


if __name__ == '__main__':
  tf.enable_eager_execution()
  ci = iter(ckpt_iter(
    '/Users/zjc/Dropbox/nlp/particularly_expensive/electra/pretraining_data/models/ngram_generator=2.ngram_pkl_path=owt_2_gram.pkl'))
