import os
import copy
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

# Test the incremental and decremental stability of everything
@torch.no_grad()
def q1t_test_radii(model, exbits_list, dataset,
                   do_save = True,
                   csv_saveto = None):
  # assert isinstance(model, MuS)
  assert isinstance(exbits_list, list)
  model.cuda().eval()
  lambd = model.lambd
  if do_save:
    assert csv_saveto is not None
    print(f"Will save to: {csv_saveto}")

  true_labels, ones_labels, exbs_labels, ones_mu_labels, exbs_mu_labels = [], [], [], [], []
  df = pd.DataFrame(columns=[
    "lambd", "p", "nnz",
    "true_label",
    "ones_label", "exbs_label", "ones_mu_label", "exbs_mu_label",
    "ones_gap", "exbs_gap", "ones_mu_gap", "exbs_mu_gap"
  ])
 
  pbar = tqdm(exbits_list)
  for i, exbits in enumerate(pbar):
    p, nnz = exbits.numel(), exbits.sum().int().item()
    exbits = exbits.view(1,-1).cuda()
    onesp = torch.ones_like(exbits)
    zerosp = torch.zeros_like(exbits)
    x, true_label = dataset[i]
    true_labels.append(true_label)
    x = x.cuda()
    xx = x.unsqueeze(0).cuda()

    x_test = torch.cat([xx, xx, xx, xx], dim=0)
    alpha_test = torch.cat([onesp, exbits, onesp, exbits], dim=0)
    mu_test = torch.cat([zerosp, zerosp, exbits, exbits], dim=0)

    y_test = model(x_test, alpha=alpha_test, mu=mu_test)
    y_ones_sorted, y_ones_order = y_test[0].sort(descending=True)
    y_exbs_sorted, y_exbs_order = y_test[1].sort(descending=True)
    y_ones_mu_sorted, y_ones_mu_order = y_test[2].sort(descending=True)
    y_exbs_mu_sorted, y_exbs_mu_order = y_test[3].sort(descending=True)

    ones_labels.append(y_ones_order[0].item())
    exbs_labels.append(y_exbs_order[0].item())
    ones_mu_labels.append(y_ones_mu_order[0].item())
    exbs_mu_labels.append(y_exbs_mu_order[0].item())
  
    this_df = pd.DataFrame({
      "lambd" : round(lambd, 4),
      "p" : p,
      "nnz" : nnz,
      "true_label" : true_labels[-1],
      "ones_label" : ones_labels[-1],
      "exbs_label" : exbs_labels[-1],
      "ones_mu_label" : ones_mu_labels[-1],
      "exbs_mu_label" : exbs_mu_labels[-1],
      "ones_gap" : (y_ones_sorted[0] - y_ones_sorted[1]).item(),
      "exbs_gap" : (y_exbs_sorted[0] - y_exbs_sorted[1]).item(),
      "ones_mu_gap" : (y_ones_mu_sorted[0] - y_ones_mu_sorted[1]).item(),
      "exbs_mu_gap" : (y_exbs_mu_sorted[0] - y_exbs_mu_sorted[1]).item(),     
    }, index=[i])

    df = pd.concat([df, this_df])
    if do_save:
      df.to_csv(csv_saveto)

    ttt = torch.tensor(true_labels)
    ones_acc = (ttt == torch.tensor(ones_labels)).float().mean().item()
    exbs_acc = (ttt == torch.tensor(exbs_labels)).float().mean().item()
    ones_mu_acc = (ttt == torch.tensor(ones_mu_labels)).float().mean().item()
    exbs_mu_acc = (ttt == torch.tensor(exbs_mu_labels)).float().mean().item()

    desc_str = f"lam {model.lambd:.4f}, "
    desc_str += f"ones ({ones_acc:.4f}, mu {ones_mu_acc:.4f}), "
    desc_str += f"exbs ({exbs_acc:.4f}, mu {exbs_mu_acc:.4f}), "
    pbar.set_description(desc_str)
  return df


def q1t_run_stuff(configs,
                  model_types = ["vit16", "resnet50", "roberta"],
                  method_types = ["shap", "lime", "vgradu", "igradu"],
                  lambds = [8/8, 4/8, 3/8, 2/8, 1/8, 1/16],
                  top_fracs = [4/8, 3/8, 2/8, 1/8],
                  patch_size = 28,
                  q = 64,
                  num_todo = None,
                  saveto_dir = None):
  assert saveto_dir is not None
  total_stuff = len(model_types) * len(method_types) * len(lambds) * len(top_fracs)
  tick = 0
  for model_type in model_types:
    for method_type in method_types:
      for lambd in lambds:
        for top_frac in top_fracs:
          tick += 1
          print(f"Running {tick}/{total_stuff}")
          # Loading 1/16
          if abs(lambd - 1/16) < 1e-5:
            model = load_model(model_type, configs["models_dir"], lambd=1/8, patch_size=patch_size, q=q)
            model.lambd = lambd
          # Otherwise load normally
          else:
            model = load_model(model_type, configs["models_dir"], lambd=lambd, patch_size=patch_size, q=q)

          exbits_list = load_exbits_list(model_type, method_type, top_frac, configs["exbits_dir"])

          if isinstance(num_todo, int):
            exbits_list = exbits_list[:num_todo]

          if model_type == "roberta":
            csv_saveto = f"q1t_{model_type}_q{q}_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
          else:
            csv_saveto = f"q1t_{model_type}_psz{patch_size}_q{q}_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
          csv_saveto = os.path.join(saveto_dir, csv_saveto)
          q1t_test_radii(model, exbits_list, configs["model2data"][model_type], csv_saveto=csv_saveto)



