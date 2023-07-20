import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import pathlib
import copy
import os
import sys
torch.manual_seed(1234)

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve().__str__()
sys.path.insert(0, BASE_DIR)

from my_models import *

# Basic device and PID setup
my_pid = os.getpid()
print(f"My PID: {my_pid}")

# setup the parser
_SAVETO_DIR = os.path.join(BASE_DIR, "saved_models")
_IMAGENET_DATA_DIR = "/data/imagenet/"
def get_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--imagenet-data-dir", type=str, default=_IMAGENET_DATA_DIR)
  parser.add_argument("-s", "--saveto-dir", type=str, default=_SAVETO_DIR)
  parser.add_argument("-e", "--epochs", type=int, default=1)
  parser.add_argument("-m", "--model-type", type=str, required=True)
  parser.add_argument("--patch-size", type=int, default=16)
  parser.add_argument("-p", "--save-prefix", type=str, default="")
  parser.add_argument("-l", "--lambds", type=float, nargs='+', required=True)
  parser.add_argument("-b", "--batch_size", type=int, default=16)
  parser.add_argument("--go", action="store_true", default=False)
  parser.add_argument("--lr", type=float, default=1e-6)
  return parser

arg_parser = get_arg_parser()
args, unknown = arg_parser.parse_known_args()

IMAGENET_TRAIN_DIR = os.path.join(args.imagenet_data_dir, "train")
IMAGENET_VAL_DIR = os.path.join(args.imagenet_data_dir, "val")
assert os.path.isdir(args.saveto_dir)
assert os.path.isdir(IMAGENET_TRAIN_DIR)
assert os.path.isdir(IMAGENET_VAL_DIR)

# The lambdas that we will be ablating
LAMBDS = [round(l,4) for l in args.lambds]

def saveto_file(epoch):
  lam_strs = [f"{l:.4f}" for l in LAMBDS]
  lam_str = "_".join(lam_strs)
  model_file = f"{args.save_prefix}{args.model_type}_durt_psz{args.patch_size}_ft__{lam_str}__epoch{epoch}.pt"
  return os.path.join(args.saveto_dir, model_file)

# Combine stuff
def binner_product(x, alpha):
  patch_size = args.patch_size
  grid_len = 224 // patch_size
  N, p = alpha.shape
  alpha = alpha.view(N,1,grid_len,grid_len).float()
  x_noised = F.interpolate(alpha, scale_factor=patch_size * 1.0) * x
  return x_noised

# Ablate an image basd on the lambdas
def ablate_image(x, num_per_prob=1):
  p = (224 // args.patch_size) ** 2
  xx = x.view(1,3,224,224).repeat(num_per_prob,1,1,1)
  prob_batches = []
  for prob in LAMBDS:
    num_hots = (p*prob + 2*torch.rand(1) - 1).round().int()
    perm = torch.randperm(p)
    alpha = torch.zeros(num_per_prob, p)
    alpha[:,perm[:num_hots]] = 1
    x_noised = binner_product(xx, alpha)
    prob_batches.append(x_noised)
  x_noiseds = torch.cat(prob_batches, dim=0)
  return x_noiseds

train_dataset = datasets.ImageFolder(
  IMAGENET_TRAIN_DIR,
  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(ablate_image),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]))

val_dataset = datasets.ImageFolder(
  IMAGENET_VAL_DIR,
  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(ablate_image),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 2)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 2)

dataloaders = { 'train': train_loader, 'val': val_loader }

@torch.no_grad()
def test_initial_loss(model, loss_fn, num_todo=10):
  model.train()
  images_processed = 0
  running_corrects = 0
  running_loss = 0.0
  pbar = tqdm(train_loader)

  # inputs: batch_size * num_ablate_samples * C * H * W
  # labels: batch_size
  for i, (inputs, labels) in enumerate(pbar):
    num_ablates = inputs.shape[1]
    inputs = inputs.flatten(0,1).cuda()
    labels = labels.view(-1,1).repeat(1,num_ablates).view(-1).cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    loss = loss_fn(outputs, labels)

    images_processed += inputs.size(0)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    avg_loss = running_loss / images_processed
    avg_acc = running_corrects.double() / images_processed
    desc_str = f"pid {my_pid}, loss {avg_loss:.4f}, acc {avg_acc:.4f}, insz {inputs.size(0)}"
    pbar.set_description(desc_str)
    if i >= num_todo:
      break

# Train the model
def train_model(model, loss_fn, num_epochs=args.epochs):
  # model = nn.DataParallel(model).cuda()
  # Test some initial loss to get a sense of later improvements
  initial_loss_todo = 200
  print(f"First testing initial loss values for num rounds: {initial_loss_todo}")
  test_initial_loss(model, loss_fn, num_todo=initial_loss_todo)

  print(f"Starting to train!")
  print(f"Will save files to: {args.saveto_dir}")
  print(f"PID: {my_pid}, lr {args.lr}")
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
      images_processed = 0

      # Iterate over data.
      pbar = tqdm(dataloaders[phase])
      # inputs: batch_size * num_ablate_samples * C * H * W
      # labels: batch_size
      for i, (inputs, labels) in enumerate(pbar):
        num_ablates = inputs.shape[1]
        inputs = inputs.flatten(0,1).cuda()
        labels = labels.view(-1,1).repeat(1,num_ablates).view(-1).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward: track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = loss_fn(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        images_processed += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        avg_loss = running_loss / images_processed
        avg_acc = running_corrects.double() / images_processed
        desc_str = f"pid {my_pid}, lambd {args.lambds}, {phase} {avg_loss:.4f}, acc {avg_acc:.4f}, insz {inputs.size(0)}"
        pbar.set_description(desc_str)

      epoch_loss = running_loss / images_processed
      epoch_acc = running_corrects.double() / images_processed
      print(f"{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

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
  torch.save(model.state_dict(), saveto_file(num_epochs))
  return model

# Set up stuff to train
if args.model_type == "vit16":
  model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
elif args.model_type == "resnet50":
  model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
else:
  print(f"Unrecognized model type: {args.model_type}")

model.cuda()
loss_fn = nn.CrossEntropyLoss(reduction='sum')

def go():
  trained_model = train_model(model, loss_fn)
  return trained_model

if args.go:
  go()

