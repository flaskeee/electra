import dataclasses
import re
from functools import singledispatch
from typing import Sequence, Union, Callable

import numpy as np
import torch
from torch import nn
import tensorflow as tf

from transformers.models.electra import modeling_electra


def load_electra(target_model: Union[str,], ckpt_path: str, discriminator_or_generator='discriminator'):
  if isinstance(target_model, str):
    target_model = modeling_electra.ElectraModel.from_pretrained(target_model)
  parameters_before_loading = {
    k: torch.clone(v)
    for k, v in target_model.named_parameters()
  }
  tf_weights = {
    k: tf.train.load_variable(ckpt_path, k)
    for k, _ in tf.train.list_variables(ckpt_path)
  }

  disc_generator_filtering_re = r'(electra/embeddings)|' + {
    'discriminator': r'(electra)',
    'generator': r'(generator)',
  }[discriminator_or_generator]
  for k, v in tf_weights.items():
    if re.match(disc_generator_filtering_re, k):
      try:
        rest_tf_names = k.split('/')[1:]
        load_weight(target_model, rest_tf_names, v)
      except NoMatchError:
        pass

  parameters_not_loaded = {
    k: v
    for k, v in target_model.named_parameters()
    if torch.equal(v, parameters_before_loading[k])
  }

  print('Parameters that are not loaded:')
  for k in parameters_not_loaded:
    print('>> didn\'t load ', k)

  return target_model


@dataclasses.dataclass
class NoMatchError(Exception):
  error_module: Union[nn.Module, nn.Parameter]
  error_rest_tf_names: Sequence[str]


def copy_np_to_torch(np_arr: np.ndarray, torch_arr: torch.tensor):
  assert isinstance(np_arr, np.ndarray)
  assert isinstance(torch_arr, torch.Tensor)

  with torch.no_grad():
    torch_arr[:] = torch.from_numpy(np_arr)

@singledispatch
def load_weight(module, rest_tf_names: Sequence[str], tf_weight: np.ndarray):
  """
  @param module: current pytorch_module to be loaded
  @param rest_tf_names: the sequence of tf (scope/variable) names that starts from the child of current module
  @return: None
  Example:
    loading the bias of key transform of a pytorch SelfAttention module
      module == SelfAttention
      full_tf_names == ['attn', 'key', 'bias']
      rest_tf_names == ['key', 'bias']
  """
  raise NotImplementedError(
    f'Not implemented for module type {type(module)}'
  )


@load_weight.register
def _load_parameter(
        torch_weight: torch.nn.Parameter,
        rest_tf_names: Sequence[str],
        tf_weight: np.ndarray,
):
  if rest_tf_names:  # found optimizer state rather than weight, e.g. gamma/adam_m
    raise NoMatchError(torch_weight, rest_tf_names)
  else:
    copy_np_to_torch(
      tf_weight,
      torch_weight,
    )


@load_weight.register
def _load_embedding(
        torch_module: torch.nn.Embedding,
        rest_tf_names: Sequence[str],
        tf_weight: np.ndarray,
):
  load_weight(torch_module.weight, rest_tf_names, tf_weight)


@load_weight.register(torch.nn.LayerNorm)
@load_weight.register(torch.nn.Linear)
def _load_builtin_layer_weight(
        torch_module: Union[torch.nn.LayerNorm, torch.nn.Linear],
        rest_tf_names: Sequence[str],
        tf_weight: np.ndarray,
):
  child_tf_name, *child_rest_tf_names = rest_tf_names
  torch_weight_name = {
    torch.nn.LayerNorm: {
      'gamma': 'weight',
      'beta': 'bias'
    },
    torch.nn.Linear: {
      'kernel': 'weight',
      'bias': 'bias'
    },
  }[type(torch_module)][child_tf_name]
  torch_weight = getattr(torch_module, torch_weight_name)
  if isinstance(torch_module, torch.nn.Linear) and child_tf_name == 'kernel':
    tf_weight = np.transpose(tf_weight)
  load_weight(torch_weight, child_rest_tf_names, tf_weight)


@load_weight.register
def electra_encoder(
        torch_module: modeling_electra.ElectraEncoder,
        rest_tf_names: Sequence[str],
        tf_weight: np.ndarray,
):
  try:
    layer_tf_name, *child_rest_tf_names = rest_tf_names
    layer_match = re.match(r'layer_([0-9]+)', layer_tf_name)
    layer_num = int(layer_match.group(1))
    child_module: modeling_electra.ElectraLayer = torch_module.layer[layer_num]
    # names are nicely aligned between all children of encoder.layer and tf encoder/layer_[n]
    #   until the builtin Linear and layerNorm
    # if the given tf weight are not part of encoder, exception will eventually rise when:
    #   all tf_names are exhausted (ValueError)
    #   no matching attribute in pytorch module (Attribute Error)
    while not isinstance(child_module, (nn.Linear, nn.LayerNorm)):
      child_tf_name, *child_rest_tf_names = child_rest_tf_names
      child_module = getattr(child_module, child_tf_name)
    load_weight(child_module, child_rest_tf_names, tf_weight)
  except (ValueError, AttributeError):
    raise NoMatchError(torch_module, rest_tf_names)


@load_weight.register(modeling_electra.ElectraModel)
def electra_model(
        torch_module: modeling_electra.ElectraModel,
        rest_tf_names: Sequence[str],
        tf_weight: np.ndarray,
):
  try:
    child_tf_name, *child_rest_tf_names = rest_tf_names
    load_weight(
      getattr(torch_module, child_tf_name),  # name as are aligned between tf and torch
      child_rest_tf_names,
      tf_weight,
    )
  except (ValueError, AttributeError):
    raise NoMatchError(torch_module, rest_tf_names)


@load_weight.register
def electra_embeddings(
        torch_module: modeling_electra.ElectraEmbeddings,
        rest_tf_names: Sequence[str],
        tf_weight: np.ndarray,
):
  try:
    child_tf_name, *child_rest_tf_names = rest_tf_names
    load_weight(
      getattr(torch_module, child_tf_name),
      child_rest_tf_names,
      tf_weight,
    )
  except (ValueError, AttributeError):
    raise NoMatchError(torch_module, rest_tf_names)


if __name__ == '__main__':
  from sys import argv
  load_electra('google/electra-small-discriminator', argv[1],)
