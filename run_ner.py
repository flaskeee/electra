import random

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import fire

import load_n2c2
import nelib.metrics
from nn_utils import script
import tensorflow as tf
import tf2pytorch


class MyTransformer(torch.nn.Module):
  def __init__(self, encoder: transformers.ElectraModel, n_labels):
    super(MyTransformer, self).__init__()
    self.encoder = encoder
    self.pred_linear = torch.nn.LazyLinear(n_labels * 3)

  def forward(self, *args, **kwargs):
    hidden = self.encoder(*args, **kwargs)[0]
    return self.pred_linear(hidden)


def main(
        weight_path,
        data_dir='',
        batch_size=64,
        seq_len=128,
        device='cuda',
        debug=False,
):
  if not torch.cuda.is_available():
    device = 'cpu'

  with torch.no_grad():
    electra = tf2pytorch.load_electra(
      "google/electra-small-discriminator",
      weight_path,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator", use_fast=True)  # Need to be changed for custom token vocab

  whole_dataset, artifacts = load_n2c2.load_ner(data_dir, tokenizer, seq_len)
  data_indices = list(range(len(whole_dataset)))
  random.shuffle(data_indices)
  train_devel_cutoff = int(len(data_indices) / 10 * 9)
  train_dataloader = DataLoader(
    torch.utils.data.Subset(whole_dataset, data_indices[:train_devel_cutoff]),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4,
  )
  devel_dataloader = DataLoader(
    torch.utils.data.Subset(whole_dataset, data_indices[train_devel_cutoff:]),
    batch_size=batch_size*4,
    shuffle=False,
    drop_last=True,
    num_workers=4,
  )

  model = MyTransformer(electra, len(artifacts.bio_converter.idx2label))

  def data2loss(model, data, aux_data):
    _, inputs, target = data
    preds = model(**inputs)
    n_labels = preds.shape[-1]
    loss = torch.nn.functional.cross_entropy(
      preds.reshape(-1, n_labels),
      target['bio_tags'].reshape(-1),
    )
    return loss

  def eval_worker(model):
    with torch.no_grad():
      positive_negative = nelib.metrics.PositiveNegative()
      for datum_idx, inputs, target in devel_dataloader:
        inputs = script.move_to_device(inputs, device)
        preds = torch.argmax(model(**inputs), dim=-1).cpu().numpy()

        for idx, datum_pred in zip(datum_idx, preds):
          datum_key = whole_dataset.idx2key(idx)
          pred_entities = artifacts.bio_converter.bio_to_offset(artifacts.word_offset[datum_key], datum_pred)
          target_entities = artifacts.concepts[datum_key]
          positive_negative.update(target_entities, pred_entities)
        if debug: break
      print(nelib.metrics.f1(**positive_negative.counter)); exit(12)



  script.train(
    model=model,
    dataloader=train_dataloader,
    data2loss=data2loss,
    eval_worker=eval_worker,
    device=device,
    debug=debug,
  )

  pass


if __name__ == '__main__':
  fire.Fire(main)
