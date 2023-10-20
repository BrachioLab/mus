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
from header import *


def q3_find_good_r(model, x, order, use_selective):
  xx = x.unsqueeze(0)
  # Binary search
  p = order.numel()
  L, R = 0, p - 1
  while L <= R:
    mid = (L + R) // 2
    # print(f"L {L}, mid {mid}, R {R}")
    exbits = torch.zeros_like(order).cuda()
    exbits[order[:mid]] = 1
    # Test if the properties holds
    consistent = check_consistent(model, x, exbits)
    cert_rs = check_cert_r(model, x, exbits)
    inc_cert_r = cert_rs[2] if use_selective else cert_rs[0]
    dec_cert_r = cert_rs[3] if use_selective else cert_rs[1]

    if consistent and inc_cert_r >= 1.0 and dec_cert_r >= 1.0:
      R = mid - 1
    else:
      L = mid + 1
  k = R + 1
  return k

# Generate certified explanations
@torch.no_grad()
def q3_find_stable_exbits(model, lambd, order_list, dataset,
                          do_save = True,
                          csv_saveto = None):
  # assert isinstance(model, MuS)
  assert len(dataset) >= len(order_list)
  assert lambd <= 0.5
  if do_save:
    assert csv_saveto is not None

  model.cuda().eval()
  ps, ks, kmus, fracs, fracmus = [], [], [], [], []
  df = pd.DataFrame(columns=["lambd", "p", "k", "k_mu"])
  pbar = tqdm(order_list)
  for i, order in enumerate(pbar):
    x, _ = dataset[i]
    x = x.cuda()
    xx = x.unsqueeze(0)

    # Binary search
    p = order.numel()
    k = q3_find_good_r(model, x, order, use_selective=False)
    k_mu = q3_find_good_r(model, x, order, use_selective=True)

    ps.append(p)
    ks.append(k)
    kmus.append(k_mu)
    fracs.append(k/p)
    fracmus.append(k_mu/p)

    avg_frac = torch.tensor(fracs).mean().item()
    avg_fracmu = torch.tensor(fracmus).mean().item()
    desc_str = f"avg_frac {avg_frac:.4f}, avg_fracmu {avg_fracmu:.4f}"
    pbar.set_description(desc_str)

    this_df = pd.DataFrame({"lambd":round(lambd,4), "p":p, "k":k, "k_mu":k_mu}, index=[i])
    df = pd.concat([df, this_df])
    if do_save:
      df.to_csv(csv_saveto)

  return df


#
def q3_run_stuff(configs,
                 model_types = ["vit16", "resnet50", "roberta"],
                 method_types = ["shap", "lime", "igradu", "vgradu"],
                 ft_lambds = [1/8., 2/8., 3/8., 4/8.],
                 q = 64,
                 patch_size = 28,
                 num_todo = None,
                 saveto_dir = None):
  assert saveto_dir is not None
  total_stuff = len(model_types) * len(method_types) * len(ft_lambds)
  tick = 0
  for model_type in model_types:
    dataset = configs["model2data"][model_type]
    for method_type in method_types:
      for lambd in ft_lambds:
        tick += 1
        print(f"Running {tick}/{total_stuff}: {model_type}, method {method_type}, lambd {lambd:.4f}")
        order_list = load_order_list(model_type, method_type, configs["exbits_dir"], q=64, patch_size=patch_size)
        if isinstance(num_todo, int) and num_todo > 0:
          order_list = order_list[:num_todo]
        model = load_model(model_type, configs["models_dir"], lambd=lambd, patch_size=patch_size, q=q)
        if model_type == "roberta":
          csv_saveto = f"q3s_{model_type}_q{q}_{method_type}_lam{lambd:.4f}.csv"
        else:
          csv_saveto = f"q3s_{model_type}_psz{patch_size}_q{q}_{method_type}_lam{lambd:.4f}.csv"
        csv_saveto = os.path.join(saveto_dir, csv_saveto)
        print(f"saveto {csv_saveto}")
        q3_find_stable_exbits(model, lambd, order_list, dataset, csv_saveto=csv_saveto)

