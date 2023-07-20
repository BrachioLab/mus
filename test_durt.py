import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as ptm
import pathlib

torch.manual_seed(1234)

from my_models import *
from qheader import *

configs = make_default_configs()


vitL1 = load_model("vit16", configs["models_dir"], lambd=1/8, q=32)
vitL2 = load_model("vit16", configs["models_dir"], lambd=2/8, q=32)
vitL4 = load_model("vit16", configs["models_dir"], lambd=4/8, q=32)
vitL8 = load_model("vit16", configs["models_dir"], lambd=8/8, q=32)


N, p, q = 2, vitL1.p, vitL1.q
xones = torch.ones(N,3,224,224)
alpha = (torch.rand(N,p) < 0.4) * 1.0
half_alpha = alpha * (torch.rand(alpha.shape) < 0.5)
vorder = torch.cat([torch.tensor(range(q)), torch.tensor(range(q))]) / q


imagenet_dataset = configs["imagenet_dataset"]
x22, l22 = imagenet_dataset[22]

