import os
import sys
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
from methods import *
from header import *


# Test stuff
def q1e_test_exbits(model, exbits_list, dataset,
                    start_ind = 0,
                    num_todo = None,
                    init_r_max_fracs = [1/8., 2/8., 3/8., 4/8.],
                    box_max_iters = -1,
                    do_box_attacks = True,
                    iter_header_msg = "",
                    do_save = True,
                    csv_saveto = None):
  assert box_max_iters > 0
  assert isinstance(exbits_list, list)
  model.cuda().eval()
  lambd = model.lambd
  if do_save:
    assert csv_saveto is not None
    print(f"Will save to: {csv_saveto}")

  true_labels = []
  ones_labels, exbs_labels, ones_mu_labels, exbs_mu_labels = [], [], [], []
  ones_gaps, exbs_gaps, ones_mu_gaps, exbs_mu_gaps = [], [], [], []
  inc_init_r_maxs, inc_curr_r_maxs, inc_max_dists, inc_num_resets, inc_num_ces = [], [], [], [], []
  dec_init_r_maxs, dec_curr_r_maxs, dec_max_dists, dec_num_resets, dec_num_ces = [], [], [], [], []

  df = pd.DataFrame(columns=[
      "data_ind",
      "lambd", "p", "nnz",
      "true_label", "ones_label", "exbs_label", "ones_mu_label", "exbs_mu_label",
      "ones_gap", "exbs_gap", "ones_mu_gap", "exbs_mu_gap",
      "inc_init_r_max", "inc_curr_r_max", "inc_max_dist", "inc_num_resets", "inc_num_ces",
      "dec_init_r_max", "dec_curr_r_max", "dec_max_dist", "dec_num_resets", "dec_num_ces",
    ])

  N = len(exbits_list)
  num_todo = N - start_ind if num_todo is None else num_todo
  todo = range(start_ind, start_ind + num_todo)
  print(f"start {todo[0]}, end {todo[-1]}, total {len(todo)}")
  assert todo[-1] <= N
  for tdi, datai in enumerate(todo):
    exbits = exbits_list[datai]
    print(f"\n{3 * '*'} PID {MY_PID}, todo {tdi+1}/{len(todo)} ({datai+1}/{N}), lam {lambd:.4f}, {iter_header_msg} {30 * '*'}")
    p, nnz = exbits.numel(), exbits.sum().int().item()
    exbits = exbits.cuda()
    x, true_label = dataset[datai]
    true_labels.append(true_label)
    x = x.cuda()
    xx = x.unsqueeze(0)

    # One-shot everything we wanna test through the model
    exbsp = exbits.view(1,p)
    onesp = torch.ones_like(exbsp)
    zerosp = torch.zeros_like(exbsp)
    x_test = torch.cat([xx, xx, xx, xx], dim=0)
    alpha_test = torch.cat([onesp, exbsp, onesp, exbsp], dim=0)
    mu_test = torch.cat([zerosp, zerosp, exbsp, exbsp], dim=0)
    y_test = model(x_test, alpha=alpha_test, mu=mu_test)
    y_sorted_values, y_sorted_order = y_test.sort(dim=1, descending=True)

    ones_label, exbs_label, ones_mu_label, exbs_mu_label = y_sorted_order[:,0]
    ones_labels.append(ones_label.item())
    exbs_labels.append(exbs_label.item())
    ones_mu_labels.append(ones_mu_label.item())
    exbs_mu_labels.append(exbs_mu_label.item())
    print(f"true {true_label}, ones ({ones_label}, mu {ones_mu_label}), exbs ({exbs_label}, mu {exbs_mu_label})")

    ones_pA, exbs_pA, ones_mu_pA, exbs_mu_pA = y_sorted_values[:,0]
    ones_pB, exbs_pB, ones_mu_pB, exbs_mu_pB = y_sorted_values[:,1]
    ones_gaps.append((ones_pA - ones_pB).item())
    exbs_gaps.append((exbs_pA - exbs_pB).item())
    ones_mu_gaps.append((ones_mu_pA - ones_mu_pB).item())
    exbs_mu_gaps.append((exbs_mu_pA - exbs_mu_pB).item())

    # As long as the explanation is non-trivial, try something
    if nnz > 0 and do_box_attacks:
      # print(f"About to do incremental box attack")
      todo_init_r_maxs = [int(p * frac) for frac in init_r_max_fracs]
      astats = find_emp_inc_stability(model, x, exbits,
                                      todo_init_r_maxs = todo_init_r_maxs,
                                      max_iters = box_max_iters,
                                      progress_bar = True)
      inc_init_r_maxs.append(-1 if astats is None else astats["init_r_max"])
      inc_curr_r_maxs.append(-1 if astats is None else astats["curr_r_max"])
      inc_max_dists.append(-1 if astats is None else astats["max_iter_dist"])
      inc_num_resets.append(-1 if astats is None else astats["num_resets"])
      inc_num_ces.append(-1 if astats is None else astats["num_ces"])

      # print(f"About to do decremental box attack")
      dstats = find_emp_dec_stability(model, x, exbits,
                                      todo_init_r_maxs = todo_init_r_maxs,
                                      max_iters = box_max_iters,
                                      progress_bar = True)
      dec_init_r_maxs.append(-1 if dstats is None else dstats["init_r_max"])
      dec_curr_r_maxs.append(-1 if dstats is None else dstats["curr_r_max"])
      dec_max_dists.append(-1 if dstats is None else dstats["max_iter_dist"])
      dec_num_resets.append(-1 if dstats is None else dstats["num_resets"])
      dec_num_ces.append(-1 if dstats is None else dstats["num_ces"])

      out_str = "box: "
      out_str += f"A (crmax {inc_curr_r_maxs[-1]}, maxd {inc_max_dists[-1]}, nces {inc_num_ces[-1]}), "
      out_str += f"D (crmax {dec_curr_r_maxs[-1]}, maxd {dec_max_dists[-1]}, nces {dec_num_ces[-1]}), "
      print(out_str)
    else:
      inc_init_r_maxs.append(-1)
      inc_curr_r_maxs.append(-1)
      inc_max_dists.append(-1)
      inc_num_resets.append(-1)
      inc_num_ces.append(-1)

      dec_init_r_maxs.append(-1)
      dec_curr_r_maxs.append(-1)
      dec_max_dists.append(-1)
      dec_num_resets.append(-1)
      dec_num_ces.append(-1)

    this_df = pd.DataFrame({
        "data_ind" : datai,
        "lambd" : round(lambd,4), "p": p, "nnz": nnz,
        "true_label" : true_labels[-1],
        "ones_label" : ones_labels[-1],
        "exbs_label" : exbs_labels[-1],
        "ones_mu_label" : ones_mu_labels[-1],
        "exbs_mu_label" : exbs_mu_labels[-1],

        "ones_gap" : ones_gaps[-1],
        "exbs_gap" : exbs_gaps[-1],
        "ones_mu_gap" : ones_mu_gaps[-1],
        "exbs_mu_gap" : exbs_mu_gaps[-1],

        "inc_init_r_max" : inc_init_r_maxs[-1],
        "inc_curr_r_max" : inc_curr_r_maxs[-1],
        "inc_max_dist" : inc_max_dists[-1],
        "inc_num_resets" : inc_num_resets[-1],
        "inc_num_ces" : inc_num_ces[-1],

        "dec_init_r_max" : dec_init_r_maxs[-1],
        "dec_curr_r_max" : dec_curr_r_maxs[-1],
        "dec_max_dist" : dec_max_dists[-1],
        "dec_num_resets" : dec_num_resets[-1],
        "dec_num_ces" : dec_num_ces[-1],
      },
      index=[datai])
    df = pd.concat([df, this_df])

    if do_save:
      df.to_csv(csv_saveto)
  return df


def q1e_run_stuff(model_type, configs,
                  lambds = [8/8, 4/8, 3/8, 2/8, 1/8],
                  method_type = "shap",
                  top_frac = 0.2500,
                  patch_size = 28,
                  q = 16,
                  start_ind = 0,
                  num_todo = 250,
                  box_max_iters = 50,
                  do_box_attacks = True,
                  saveto_dir = None):
  dataset = configs["model2data"][model_type]
  exbits_list = configs["model2exbits"][model_type]

  for i, lambd in enumerate(lambds):
    model = load_model(model_type, configs["models_dir"], lambd=lambd, patch_size=patch_size, q=q)
    if model_type == "roberta":
      csv_saveto = f"q1e_{model_type}_q{model.q}_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
    else:
      csv_saveto = f"q1e_{model_type}_psz{patch_size}_q{model.q}_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
    csv_saveto = os.path.join(saveto_dir, csv_saveto)
    header_msg = f"{model_type}"
    print(f"Launching: {csv_saveto}")
    q1e_test_exbits(model, exbits_list, dataset,
                    box_max_iters = box_max_iters,
                    csv_saveto = csv_saveto,
                    iter_header_msg = header_msg,
                    do_box_attacks = do_box_attacks,
                    start_ind = start_ind,
                    num_todo = num_todo)


if __name__ == "__main__":
  configs = make_default_configs()
  configs["saveto_dir"] = os.path.join(configs["dump_dir"], "q1_boxatk")

  method_type, top_frac = "shap", 0.25
  vit16_exbits_list = load_exbits_list("vit16", method_type, top_frac, configs["exbits_dir"])
  resnet50_exbits_list = load_exbits_list("resnet50", method_type, top_frac, configs["exbits_dir"])
  roberta_exbits_list = load_exbits_list("roberta", method_type, top_frac, configs["exbits_dir"])

  configs["model2exbits"] = {
    "vit16" : vit16_exbits_list,
    "resnet50" : resnet50_exbits_list,
    "roberta" : roberta_exbits_list
  }

  assert os.path.isdir(configs["saveto_dir"])

# run_stuff("vit16", configs)

