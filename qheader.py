import os
import sys
import copy
import pathlib
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torchvision
import torchvision.models as pt_models
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import math

torch.manual_seed(1234)

from my_models import *
from my_datasets import *
from methods import *

#
MY_PID = os.getpid()
print(f"PID: {MY_PID}")

# Default arguments for argparse
_BASE_DIR = pathlib.Path(__file__).parent.resolve()
_EXBITS_DIR = os.path.join(_BASE_DIR, "exbits")
_MODELS_DIR = os.path.join(_BASE_DIR, "saved_models")
_IMAGENET_DATA_DIR = "/data/imagenet"
_TWEET_DATA_DIR = os.path.join(_BASE_DIR, "tweeteval/datasets/sentiment")

# Call this
def make_default_configs():
  parser = argparse.ArgumentParser()
  parser.add_argument("--models-dir", type=str, default=_MODELS_DIR)
  parser.add_argument("--exbits-dir", type=str, default=_EXBITS_DIR)
  parser.add_argument("--imagenet-data-dir", type=str, default=_IMAGENET_DATA_DIR)
  parser.add_argument("--tweet-data-dir", type=str, default=_TWEET_DATA_DIR)
  args, unknown = parser.parse_known_args()

  # Parse stuff and ensure directory structure
  models_dir = args.models_dir
  exbits_dir = args.exbits_dir
  assert os.path.isdir(models_dir)
  assert os.path.isdir(exbits_dir)

  imagenet_val_dir = os.path.join(args.imagenet_data_dir, "val")
  tweet_val_dir = args.tweet_data_dir
  tweet_val_text = os.path.join(tweet_val_dir, "val_text.txt")
  tweet_val_labels = os.path.join(tweet_val_dir, "val_labels.txt")
  assert os.path.isdir(imagenet_val_dir)
  assert os.path.isfile(tweet_val_text) and os.path.isfile(tweet_val_labels)

  # Attempt to load randperm if they exist, otherwise randomly generate
  raw_imagenet_dataset = datasets.ImageFolder(
      imagenet_val_dir,
      transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
  imagenet_randperm_file = os.path.join(args.exbits_dir, "imagenet_randperm.pt")
  if os.path.isfile(imagenet_randperm_file):
    imagenet_randperm = torch.load(imagenet_randperm_file)
  else:
    print("WARNING: new imagenet randperm used")
    torch.manual_seed(1234)
    imagenet_randperm = torch.randperm(len(raw_imagenet_dataset))
  imagenet_dataset = Subset(raw_imagenet_dataset, indices=imagenet_randperm)

  raw_tweet_dataset = TweetDataset(tweet_val_text, tweet_val_labels)
  tweet_randperm_file = os.path.join(args.exbits_dir, "tweet_randperm.pt")
  if os.path.isfile(tweet_randperm_file):
    tweet_randperm = torch.load(tweet_randperm_file)
  else:
    print("WARNING: new tweet randperm used")
    torch.manual_seed(1234)
    tweet_randperm = torch.randperm(len(raw_tweet_dataset))
  tweet_dataset = Subset(raw_tweet_dataset, indices=tweet_randperm)

  # Some useful dicts
  return {
    "base_dir" : _BASE_DIR,
    "models_dir" : models_dir,
    "exbits_dir" : exbits_dir,
    "imagenet_val_dir" : imagenet_val_dir,
    "tweet_val_dir" : tweet_val_dir,

    "imagenet_dataset" : imagenet_dataset,
    "tweet_dataset" : tweet_dataset,

    "model2data" : {
      "vit16" : imagenet_dataset,
      "resnet50" : imagenet_dataset, 
      "roberta" : tweet_dataset
    },

    "model2method" : {
      "lime" : LimeExplainer(0.25),
      "shap" : GradShapExplainer(0.25),
      "vgrads" : PlainGradExplainer(0.25, signed=True),
      "vgradu" : PlainGradExplainer(0.25, signed=False),
      "igrads" : IntGradExplainer(0.25, signed=True),
      "igradu" : IntGradExplainer(0.25, signed=False),}
  }


# Load baseline / fine-tuned models quickly
def load_model(model_type, models_dir, lambd=None, ft_epoch=5, patch_size=28, q=64, verbose=True):
  assert model_type in ["resnet50", "vit16", "roberta"]
  assert os.path.isdir(models_dir)
  roberta_url = "cardiffnlp/twitter-roberta-base-sentiment"
  if lambd is None:
    if model_type == "resnet50":
      _resnet50 = pt_models.resnet50(weights=pt_models.ResNet50_Weights.IMAGENET1K_V1)
      return MuSImageNet(MyResNet(_resnet50), patch_size=patch_size, q=q, lambd=1.0).eval().cpu()
    if model_type == "vit16":
      _vit16 = pt_models.vit_b_16(weights=pt_models.ViT_B_16_Weights.IMAGENET1K_V1)
      return MuSImageNet(MyViT(_vit16), patch_size=patch_size, q=q, lambd=1.0).eval().cpu()
    if model_type == "roberta":
      _roberta = RobertaForSequenceClassification.from_pretrained(roberta_url)
      return MuSTweet(MyRoberta(_roberta), q=q, lambd=1.0).eval().cpu()
  else:
    # Otherwise if we're not at an exact multiple of 1/8, round up to the next closest one
    ft_lambd = math.ceil(8 * lambd) / 8
    if model_type == "roberta":
      ft_model_file = f"{model_type}_durt_ft__{ft_lambd:.4f}_{ft_lambd/2:.4f}__epoch{ft_epoch}.pt"
    else:
      ft_model_file = f"{model_type}_durt_psz{patch_size}_ft__{ft_lambd:.4f}_{ft_lambd/2:.4f}__epoch{ft_epoch}.pt"

    if verbose: print(f"loading: {ft_model_file}")

    state_dict = torch.load(os.path.join(models_dir, ft_model_file), map_location="cpu")
    if model_type == "resnet50":
      ft_model = pt_models.resnet50()
      ft_model.load_state_dict(state_dict)
      return MuSImageNet(MyResNet(ft_model), patch_size=patch_size, q=q, lambd=lambd).eval().cpu()
    if model_type == "vit16":
      ft_model = pt_models.vit_b_16()
      ft_model.load_state_dict(state_dict)
      return MuSImageNet(MyViT(ft_model), patch_size=patch_size, q=q, lambd=lambd).eval().cpu()
    if model_type == "roberta":
      ft_model = RobertaForSequenceClassification.from_pretrained(roberta_url)
      ft_model.load_state_dict(state_dict)
      return MuSTweet(MyRoberta(ft_model), q=q, lambd=lambd).eval().cpu()


# Load exbits
def load_exbits_list(model_type, method_type, top_frac, exbits_dir, q=64, patch_size=28):
  assert os.path.isdir(exbits_dir)
  if model_type == "roberta":
    exbits_file = f"{model_type}_{method_type}_q{q}__exbtop{top_frac:.4f}.pt"
  else:
    exbits_file = f"{model_type}_{method_type}_q{q}_psz{patch_size}__exbtop{top_frac:.4f}.pt"
  exbits_file = os.path.join(exbits_dir, exbits_file)
  return torch.load(exbits_file)

# Load order
def load_order_list(model_type, method_type, exbits_dir, q=64, patch_size=28):
  assert os.path.isdir(exbits_dir)
  if model_type == "roberta":
    exbits_file = f"{model_type}_{method_type}_q{q}__order.pt"
  else:
    exbits_file = f"{model_type}_{method_type}_q{q}_psz{patch_size}__order.pt"
  exbits_file = os.path.join(exbits_dir, exbits_file)
  return torch.load(exbits_file)


# Generate explanations if needed
def generate_model_method_explanations(model_type, method_type, configs,
                                       patch_size = 28,
                                       q = 64,
                                       lambd = 1.0,
                                       top_fracs = [0.125, 0.25, 0.375, 0.5],
                                       seed = 1234,
                                       num_todo = -1,
                                       do_save = True):
  model = load_model(model_type, configs["models_dir"], lambd=lambd).cuda().eval()
  model.use_voting = False  # This allows for gradients

  dataset = configs["model2data"][model_type]
  assert len(dataset) >= num_todo and num_todo >= 1
  if do_save:
    if model_type == "roberta":
      order_saveto = f"{model_type}_{method_type}_q{q}__order.pt"
    else:
      order_saveto = f"{model_type}_{method_type}_q{q}_psz{patch_size}__order.pt"
    order_saveto = os.path.join(configs["exbits_dir"], order_saveto)
    print(f"Will save to: {order_saveto}")

  explainer = configs["model2method"][method_type]
  all_orders = []
  pbar = tqdm(range(num_todo))
  for i in pbar:
    x, _ = dataset[i]
    x = x.cuda()
    _, order = explainer.find_explanation(model, x, get_order=True)
    all_orders.append(order.detach().cpu())

  if do_save:
    torch.save(all_orders, order_saveto)

  for p in top_fracs:
    all_exbits = []
    for order in all_orders:
      exbits = torch.zeros_like(order)
      k = math.ceil(order.numel() * p)
      exbits[order[:k]] = 1
      all_exbits.append(exbits)
    if do_save:
      if model_type == "roberta":
        exbits_saveto = f"{model_type}_{method_type}_q{q}__exbtop{p:.4f}.pt"
      else:
        exbits_saveto = f"{model_type}_{method_type}_q{q}_psz{patch_size}__exbtop{p:.4f}.pt"
      exbits_saveto = os.path.join(configs["exbits_dir"], exbits_saveto)
      torch.save(all_exbits, exbits_saveto)
  return all_orders


# Explanations for different things
def generate_vit16_explanations(configs, **kwargs):
  generate_model_method_explanations("vit16", "shap", configs, **kwargs)
  generate_model_method_explanations("vit16", "vgradu", configs, **kwargs)
  generate_model_method_explanations("vit16", "igradu", configs, **kwargs)
  generate_model_method_explanations("vit16", "lime", configs, **kwargs)


def generate_resnet50_explanations(configs, **kwargs):
  generate_model_method_explanations("resnet50", "shap", configs, **kwargs)
  generate_model_method_explanations("resnet50", "vgradu", configs, **kwargs)
  generate_model_method_explanations("resnet50", "igradu", configs, **kwargs)
  generate_model_method_explanations("resnet50", "lime", configs, **kwargs)


def generate_roberta_explanations(configs, **kwargs):
  generate_model_method_explanations("roberta", "shap", configs, **kwargs)
  generate_model_method_explanations("roberta", "vgradu", configs, **kwargs)
  generate_model_method_explanations("roberta", "igradu", configs, **kwargs)
  generate_model_method_explanations("roberta", "lime", configs, **kwargs)


if __name__ == "__main__":
  configs = make_default_configs()

