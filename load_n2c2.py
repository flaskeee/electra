import dataclasses
from collections import defaultdict
from functools import reduce
from itertools import accumulate
from pathlib import Path
import re  # This tricks PyCharm into using RE highlighting on regex
import regex as re
import torch

from nelib.convert import BIOConverter
from nelib.definitions import Entity
from typing import Dict, Tuple, List


class DictDataset:
  def __init__(self, *dicts):
    self._dicts = dicts
    self._valid_keys = [
      k for k in self._dicts[0]
      if all(k in d for d in self._dicts)
    ]

  def __len__(self):
    return len(self._valid_keys)

  def __getitem__(self, idx):
    key = self.idx2key(idx)
    return (idx,) + tuple(
      d[key] for d in self._dicts
    )

  def idx2key(self, idx):
    return self._valid_keys[idx]


datum_idx_t = Tuple[str, int]


@dataclasses.dataclass
class Artifacts:
  text: Dict[datum_idx_t, str]
  concepts: Dict[datum_idx_t, List[Entity]]
  word_offset: Dict[datum_idx_t, List[Tuple[int, int]]]
  bio_converter: BIOConverter


def load_ner(data_dir, tokenizer, seq_len) -> (DictDataset, Artifacts):
  data_dir = Path(data_dir)

  concept_re = re.compile(
    r'c="(?P<text>.*)" (?P<l_line>\d+):(?P<l_word>\d+) (?P<r_line>\d+):(?P<r_word>\d+)\|\|t="(?P<label>.*)"'
  )
  concept_dict: Dict[datum_idx_t, List[Entity]] = defaultdict(list)  # indexed by (doc_id, line_no)
  concept_labels = set()
  for f in data_dir.glob('**/*.con'):
    doc_id = f.name[:-4]
    for line in open(f):
      match = re.match(concept_re, line)
      line_no = int(match.group('l_line'))
      concept_labels.add(match.group('label'))
      concept_dict[(doc_id, line_no)].append(
        Entity(
          l=int(match.group('l_word')),
          r=int(match.group('r_word')),
          label=match.group('label'),
          doc=str((doc_id, line_no)),
          text=match.group('text'),
        )
      )
  print('all concepts loaded')

  bio_converter = BIOConverter(labels=concept_labels)
  input_dict: Dict[datum_idx_t, Dict] = {}
  output_dict: Dict[datum_idx_t, Dict] = {}
  text_dict: Dict[datum_idx_t, str] = {}
  wordoffset_dict: Dict[datum_idx_t, List[Tuple[int, int]]] = {}
  id_offset_increment = {
    tk_id: 0 if tk.startswith('##') or re.match(r'\[.*]', tk) else 1
    for tk, tk_id in tokenizer.get_vocab().items()
  }
  for f in data_dir.glob('**/*.txt'):
    doc_id = f.name[:-4]
    for line_no, line in enumerate(open(f)):
      if len(line) < 10:
        continue

      datum_key = (doc_id, line_no)
      text_dict[datum_key] = line
      input_dict[datum_key] = {
        k: torch.squeeze(v, dim=0)
        for k, v in
        tokenizer(line, max_length=seq_len, truncation=True, padding='max_length', return_tensors='pt').items()
      }

      word_offset = [(-1, 0)]
      for curr_id in input_dict[datum_key]['input_ids'].numpy()[1:]:
        l, r = word_offset[-1]
        if id_offset_increment[curr_id]:
          l += 1
          r += 1
        word_offset.append((l, r))
      '''
      word_offset = list(accumulate(
        input_dict[datum_key]['input_ids'],
        lambda prev_offset, curr_id: (prev_offset[0] + id_offset_increment[curr_id], prev_offset[1] + id_offset_increment[curr_id]),
        initial=(-1, 0),
      ))
      '''
      wordoffset_dict[datum_key] = word_offset

      output_dict[datum_key] = {
        'bio_tags': torch.tensor(bio_converter.offset_to_bio(word_offset, concept_dict[datum_key]))
      }
      assert len(word_offset) == 128
      assert len(output_dict[datum_key]['bio_tags']) == 128
  print('all data loaded')
  artifacts = Artifacts(
    text=text_dict,
    concepts=concept_dict,
    word_offset=wordoffset_dict,
    bio_converter=bio_converter,
  )
  return DictDataset(input_dict, output_dict), artifacts
