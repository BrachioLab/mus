import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from .mus import *

roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_tokenizer_fn = lambda text: roberta_tokenizer(text, return_tensors='pt')['input_ids']

# Roberta that just returns the tensor output
# negative (0), neutral (1), positive (2)
class MyRoberta(nn.Module):
  def __init__(self, roberta,
                     tokenizer_fn = roberta_tokenizer_fn):
    super(MyRoberta, self).__init__()
    assert isinstance(roberta, RobertaForSequenceClassification)
    self.roberta = copy.deepcopy(roberta)
    self.embedding_fn = self.roberta.get_input_embeddings()
    assert callable(tokenizer_fn)
    self.tokenizer_fn = tokenizer_fn

  def tokenize(self, text):
    return self.tokenizer_fn(text)

  def embed_token_ids(self, tokids):
    return self.embedding_fn(tokids)

  # Toggles embedding or token mode; catch but do no forward kwargs
  def forward(self, x_or_tokids, **kwargs):
    if x_or_tokids.dtype == torch.int64:
      y = self.roberta(x_or_tokids)
    else:
      y = self.roberta(inputs_embeds=x_or_tokids.float())
    return y.logits

# Wrapper for a mus language model
class MuSTweet(MuS):
  def __init__(self, base_model, q, lambd,
                     embedding_fn = None,
                     mask_tokid = roberta_tokenizer.mask_token_id,
                     bos_tokid = roberta_tokenizer.bos_token_id,
                     eos_tokid = roberta_tokenizer.eos_token_id,
                     tokenizer_fn = roberta_tokenizer_fn):
    super(MuSTweet, self).__init__(base_model, q=q, lambd=lambd)
    embedding_fn = self.base_model.embedding_fn if embedding_fn is None else embedding_fn
    assert callable(embedding_fn)
    self.embedding_fn = embedding_fn
    self.mask_tokid = mask_tokid
    self.mask_tokid_pt = torch.Tensor([mask_tokid]).view(1,1).long()
    self.bos_tokid = bos_tokid
    self.bos_tokid_pt = torch.Tensor([bos_tokid]).view(1,1).long()
    self.eos_tokid = eos_tokid
    self.eos_tokid_pt = torch.Tensor([eos_tokid]).view(1,1).long()
    assert callable(tokenizer_fn)
    self.tokenizer_fn = tokenizer_fn

  def tokenize(self, text):
    return self.tokenizer_fn(text)

  def embed_token_ids(self, tokids):
    return self.embedding_fn(tokids)

  # Cut out the bos and eos tokens
  def alpha_shape(self, x_or_tokids):
    N, L = x_or_tokids.shape[:2]
    return torch.Size([N, L-2])

  # Binner product on the embedding space
  def binner_product(self, x, alpha):
    N, L, _ = x.shape
    assert alpha.shape == torch.Size([N, L-2])
    alplus = F.pad(alpha, (1,1), "constant", 1).view(N,L,1) # [1; alpha; 1]
    mask_tokid_pt = torch.Tensor([self.mask_tokid]).long()
    mask_embed = self.embedding_fn(self.mask_tokid_pt.to(x.device)).detach()
    mask = (alplus == 0) * mask_embed.to(x.device)
    x_noised = alplus*x + mask
    return x_noised

  # Forward function can take tokens or string, or strings
  def forward(self, tokids_or_str, tokenize_str=False, **kwargs):
    if isinstance(tokids_or_str, torch.Tensor):
      todo_tokids = [tokids_or_str]
    elif isinstance(tokids_or_str, str):
      todo_tokids = [self.tokenizer_fn(tokids_or_str)]
    elif isinstance(tokids_or_str, list) and all([isinstance(t, torch.Tensor) for t in tokids_or_str]):
      todo_tokids = tokids_or_str
    elif isinstance(tokids_or_str, list) and all([isinstance(s, str) for s in tokids_or_str]):
      todo_tokids = [self.tokenizer_fn(s) for s in tokids_or_str]
    else:
      raise NotImplementedError(f"unrecognized: {tokids_or_str}")

    all_y = []
    for tokids in todo_tokids:
      N, _ = tokids.shape
      assert (tokids[:,0] == self.bos_tokid).sum() == N
      assert (tokids[:,-1] == self.eos_tokid).sum() == N
      x = self.embed_token_ids(tokids)
      y = super(MuSTweet, self).forward(x, **kwargs)
      all_y.append(y)
    all_y = torch.cat(all_y, dim=0)
    return all_y



