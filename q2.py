import os
import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
import math
import pathlib

torch.manual_seed(1234)

from my_models import *
from qheader import *

# Run a bunch of samples
@torch.no_grad()
def q2_test_radii(model, dataset,
                  num_todo = -1,
                  do_save = True,
                  csv_saveto = None):
  assert len(dataset) >= num_todo and num_todo >= 1
  assert isinstance(model, MuS)
  model.cuda().eval()
  model.use_voting = True
  lambd = model.lambd
  if do_save:
    assert csv_saveto is not None
    print(f"Will save to: {csv_saveto}")

  true_labels, ones_labels, logit_gaps, cert_rs = [], [], [], []

  df = pd.DataFrame(columns=[
    'lambd', 'p', 'true_label', 'ones_label', 'logit_gap', 'cert_r'
  ])

  pbar = tqdm(range(num_todo))
  for i in pbar:
    x, true_label = dataset[i]
    true_labels.append(true_label)
    xx = x.unsqueeze(0).cuda()
    alpha = torch.ones(model.alpha_shape(xx)).to(xx.device)
    y = model(xx, alpha=alpha)[0]
    y_sorted, y_order = y.sort(descending=True)
    logit_gap = (y_sorted[0] - y_sorted[1]).item()
    cert_r = logit_gap / (2 * lambd)
    ones_label = y_order[0].item()
    ones_labels.append(ones_label)

    this_df = pd.DataFrame({
      'lambd' : round(lambd, 4),
      'p' : alpha.numel(),
      'true_label' : true_label,
      'ones_label' : ones_label,
      'logit_gap' : logit_gap,
      'cert_r' : cert_r,
    }, index=[i])

    df = pd.concat([df, this_df])
    if do_save:
      df.to_csv(csv_saveto)

    acc = (torch.tensor(true_labels) == torch.tensor(ones_labels)).float().mean().item()
    desc_str = f"acc {acc:.4f}"
    pbar.set_description(desc_str)

  return df


all_lambds = [8/8, 7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8]

# Run the thing
def q2_run_stuff(configs,
                 model_types = ["vit16", "resnet50", "roberta"],
                 lambds = [8/8., 4/8., 3/8., 2/8., 1/8.],
                 patch_size = 28,
                 q = 64,
                 num_todo = 2000):
  assert num_todo > 0
  total_stuff = len(model_types) * len(lambds)
  tick = 0
  for model_type in model_types:
    dataset = configs["model2data"][model_type]
    for lambd in lambds:
      tick += 1
      print(f"Running {tick}/{total_stuff}")
      model = load_model(model_type, configs["models_dir"], lambd=lambd, patch_size=patch_size, q=q)
      if model_type == "roberta":
        csv_saveto = f"q2_{model_type}_q{q}_lam{model.lambd:.4f}.csv"
      else:
        csv_saveto = f"q2_{model_type}_psz{patch_size}_q{q}_lam{model.lambd:.4f}.csv"
      csv_saveto = os.path.join(configs["saveto_dir"], csv_saveto)
      q2_test_radii(model, dataset, num_todo=num_todo, csv_saveto=csv_saveto)

      lambd_half = lambd - 1/16.
      model.lambd = lambd_half
      if model_type == "roberta":
        csv_half_saveto = f"q2_{model_type}_q{q}_lam{model.lambd:.4f}.csv"
      else:
        csv_half_saveto = f"q2_{model_type}_psz{patch_size}_q{q}_lam{model.lambd:.4f}.csv"
      csv_half_saveto = os.path.join(configs["saveto_dir"], csv_half_saveto)
      q2_test_radii(model, dataset, num_todo=num_todo, csv_saveto=csv_half_saveto)


if __name__ == "__main__":
  configs = make_default_configs()
  configs["saveto_dir"] = os.path.join(configs["base_dir"], "dump", "q2")
  assert os.path.isdir(configs["saveto_dir"])

# run_stuff(configs, model_types=["vit16"], lambds=[1/8, 2/8])

