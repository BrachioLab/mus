import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models as torch_models
from torchvision import transforms

from .mus import *

# Resnet with a normalization prepend
class MyResNet(nn.Module):
  def __init__(self, resnet):
    super(MyResNet, self).__init__()
    # assert isinstance(resnet, torch_models.resnet.ResNet)
    self.resnet = resnet
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  # Catch but do not forward kwargs
  def forward(self, x, **kwargs):
    x = x.clamp(0,1)
    x = self.normalize(x)
    y = self.resnet(x)
    return y

# ViT with a normalization prepend
class MyViT(nn.Module):
  def __init__(self, vit):
    super(MyViT, self).__init__()
    # assert isinstance(vit, torch_models.vision_transformer.VisionTransformer)
    self.vit = vit
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  # Catch but do not forward kwargs
  def forward(self, x, **kwargs):
    x = x.clamp(0,1)
    x = self.normalize(x)
    y = self.vit(x)
    return y

# MuS wrapper around vision models (namely resnet and vit)
class MuSImageNet(MuS):
  def __init__(self, base_model, patch_size, q, lambd,
               image_shape = torch.Size([3,224,224])):
    super(MuSImageNet, self).__init__(base_model, q=q, lambd=lambd)
    assert len(image_shape) == 3
    assert image_shape[1] % patch_size == 0
    assert image_shape[2] % patch_size == 0
    self.patch_size = patch_size
    self.image_shape = image_shape

    self.grid_h_len = image_shape[1] // patch_size
    self.grid_w_len = image_shape[2] // patch_size
    self.p = self.grid_h_len * self.grid_w_len

  # x: (N,C,H,W), alpha: (N,p)
  def alpha_shape(self, x):
    return torch.Size([x.size(0), self.p])

  def binner_product(self, x, alpha):
    N, p = alpha.shape
    alpha = alpha.view(N,1,self.grid_h_len,self.grid_w_len).float()
    x_noised = F.interpolate(alpha, scale_factor=self.patch_size * 1.0) * x
    return x_noised

  # Have custom forward thing for doing the normalization
  def forward(self, x, **kwargs):
    y = super(MuSImageNet, self).forward(x, **kwargs)
    return y

