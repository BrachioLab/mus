import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib

import argparse
import linecache
import copy
import time
import random
import os
import sys
torch.manual_seed(1234)

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve().__str__()
sys.path.insert(0, BASE_DIR)

from my_models import *
from my_datasets import TweetDataset, tweet_tokenizer, tweet_tokenizer_fn

# Basic device and PID estup
my_pid = os.getpid()
print(f"My PID: {my_pid}")

# Set up the parser
_SAVETO_DIR = os.path.join(BASE_DIR, "saved_models")
_TWEET_DATA_DIR = os.path.join(BASE_DIR, "../tweeteval/datasets/sentiment")
def get_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--tweet-data-dir", type=str, default=_TWEET_DATA_DIR)
  parser.add_argument("-s", "--saveto-dir", type=str, default=_SAVETO_DIR)
  parser.add_argument("-e", "--epochs", type=int, default=1)
  parser.add_argument("-m", "--model-type", type=str, default="roberta")
  parser.add_argument("-p", "--save-prefix", type=str, default="")
  parser.add_argument("-l", "--lambdas", nargs='+', required=True)
  parser.add_argument("--go", action="store_true", default=False)
  parser.add_argument("--lr", type=float, default=1e-6)
  return parser

arg_parser = get_arg_parser()
args, unknown = arg_parser.parse_known_args()

TRAIN_TEXT = os.path.join(args.tweet_data_dir, "train_text.txt")
TRAIN_LABELS = os.path.join(args.tweet_data_dir, "train_labels.txt")
VAL_TEXT = os.path.join(args.tweet_data_dir, "val_text.txt")
VAL_LABELS = os.path.join(args.tweet_data_dir, "val_labels.txt")
assert os.path.isdir(args.saveto_dir)
assert os.path.isfile(TRAIN_TEXT) and os.path.isfile(TRAIN_LABELS)
assert os.path.isfile(VAL_TEXT) and os.path.isfile(VAL_LABELS)

# The lambdas that we will be ablating
LAMBDS = [round(float(l), 4) for l in args.lambdas]

# Where to save, depending on the epoch
def saveto_file(epoch):
  lam_strs = [f"{l:.4f}" for l in LAMBDS]
  lam_str = "_".join(lam_strs)
  model_file = f"{args.save_prefix}{args.model_type}_durt_ft__{lam_str}__epoch{epoch}.pt"
  return os.path.join(args.saveto_dir, model_file)

# Ablate the tokens base
def binner_product(tokids, alpha):
  assert tokids.ndim == 1
  L = tokids.size(0)
  assert alpha.shape == torch.Size([L-2])
  alplus = F.pad(alpha, (1,1), "constant", 1) # [1; alpha; 1]
  mask = (alplus == 0) * tweet_tokenizer.mask_token_id
  toks_noised = tokids * alplus + mask
  return toks_noised.long()

def ablate_token_ids(tokids, num_per_prob=1):
  assert tokids.ndim == 1
  assert tokids[0] == tweet_tokenizer.bos_token_id
  assert tokids[-1] == tweet_tokenizer.eos_token_id
  L = tokids.size(0)
  alphlen = L - 2
  prob_batches = []
  for prob in LAMBDS:
    num_hots = (alphlen*prob + 2*torch.rand(1) - 1).round().int()
    perm = torch.randperm(alphlen)
    alpha = torch.zeros(alphlen).cuda()
    alpha[perm[:num_hots]] = 1
    tokids_noised = binner_product(tokids, alpha)
    prob_batches.append(tokids_noised.view(1,L))
  inputs = torch.cat(prob_batches, dim=0)
  return inputs

# Datasets
train_dataset = TweetDataset(TRAIN_TEXT, TRAIN_LABELS)
val_dataset = TweetDataset(VAL_TEXT, VAL_LABELS)

train_loader = DataLoader(
  train_dataset,
  batch_size = 1,
  shuffle = True,
  num_workers = 1)

val_loader = DataLoader(
  val_dataset,
  batch_size = 1,
  shuffle = True,
  num_workers = 1)

dataloaders = { 'train': train_loader, 'val': val_loader }

# Some initial testing of the loss value
@torch.no_grad()
def test_initial_loss(model, loss_fn, num_todo=100):
  model.train()
  tweet_processed = 0
  running_corrects = 0
  running_loss = 0.0
  pbar = tqdm(train_loader)

  # inputs: (1,L), labels: 1
  for i, (tokids, label) in enumerate(pbar):
    assert tokids.size(0) == 1
    tokids = tokids.cuda()[0]
    inputs = ablate_token_ids(tokids)
    num_ablates, L = inputs.shape
    labels = label.cuda().repeat(num_ablates)
    outputs = model(inputs)
    logits = outputs.logits
    _, preds = torch.max(logits, 1)
    loss = loss_fn(logits, labels)

    tweet_processed += inputs.size(0)
    running_loss += loss.item() * num_ablates
    running_corrects += torch.sum(preds == labels)
    avg_loss = running_loss / tweet_processed
    avg_acc = running_corrects.double() / tweet_processed
    desc_str = f"pid {my_pid}, loss {avg_loss:.4f}, acc {avg_acc:.4f}"
    pbar.set_description(desc_str)
    if i >= num_todo:
      break

# Train the model
def train_model(model, loss_fn, num_epochs=args.epochs):
  # model = nn.DataParallel(model).cuda()
  # Test some initial loss to get a sense of later improvements
  initial_loss_todo = 400
  print(f"First testing initial loss values for num rounds: {initial_loss_todo}")
  test_initial_loss(model, loss_fn, num_todo=initial_loss_todo)

  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  print(f"Starting to train! PID: {my_pid}, lr {args.lr}")
  print(f"Will save files to: {args.saveto_dir}")
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(1, num_epochs+1):
    saveto = saveto_file(epoch)
    print(f'Epoch {epoch}/{num_epochs} - will save to: {saveto}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0
      tweet_processed = 0

      # Iterate over data.
      pbar = tqdm(dataloaders[phase])
      # token_ids: (1,L), labels: 1
      for i, (tokids, label) in enumerate(pbar):
        tokids = tokids.cuda()[0]
        inputs = ablate_token_ids(tokids)
        num_ablates, _ = inputs.shape
        labels = label.cuda().repeat(num_ablates)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward: track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          logits = outputs.logits
          loss = loss_fn(logits, labels)
          _, preds = torch.max(logits, 1)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        tweet_processed += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
        avg_loss = running_loss / tweet_processed
        avg_acc = running_corrects.double() / tweet_processed
        desc_str = f"pid {my_pid}, lambd {args.lambdas}, {phase} loss {avg_loss:.4f}, acc {avg_acc:.4f}"
        pbar.set_description(desc_str)

      epoch_loss = running_loss / tweet_processed
      epoch_acc = running_corrects.double() / tweet_processed
      print(f"{phase} loss {epoch_loss:.4f}, acc {epoch_acc:.4f}")

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), saveto)

    print()

  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val acc {best_acc:4f}')

  # load best model weights
  model.load_state_dict(best_model_wts)
  torch.save(model.state_dict(), saveto)
  return model

# Set up stuff to train
if args.model_type == "roberta":
  model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
else:
  print(f"Unrecognized model type: {args.model_type}")

model.cuda()
loss_fn = nn.CrossEntropyLoss(reduction='sum')

def go():
  trained_model = train_model(model, loss_fn)

if args.go:
  go()
