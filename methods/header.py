import numpy as np
import math
import torch
from tqdm import tqdm

# Superclass for every explanation method
class Explainer:
  # Everything needs to override this
  def find_explanation(self, model, x, **kwargs):
    raise NotImplementedError

# Just return the vector of ones
class OnesExplainer(Explainer):
  def __init__(self):
    super(OnesExplainer, self).__init__()

  def __str__(self):
    return "ones"

  def find_explanation(self, model, x, get_order=False):
    x = x.unsqueeze(0)
    exbits = torch.ones(model.alpha_shape(x)).to(x.device)
    if get_order:
      return exbits, torch.tensor(range(0, exbits.numel()))
    else:
      return exbits

# Keep guessing until something works ... or just give up
class RandomExplainer(Explainer):
  def __init__(self, hot_prob, num_tries=50):
    super(RandomExplainer, self).__init__()
    self.hot_prob = hot_prob
    self.num_tries = num_tries

  def __str__(self):
    return f"rand{self.hot_prob:.4f}"

  @torch.no_grad()
  def find_explanation(self, model, x,
                       hot_prob = None,
                       num_tries = None,
                       get_order = False,
                       progress_bar = False,
                       **kwargs):
    model.eval()
    xx = x.unsqueeze(0)
    hot_prob = self.hot_prob if hot_prob is None else hot_prob
    alpha_shape = model.alpha_shape(xx)
    k = math.ceil(alpha_shape.numel() * hot_prob)
    target_label = model(xx)[0].argmax()
    num_tries = self.num_tries if num_tries is None else num_tries
    pbar = tqdm(range(num_tries)) if progress_bar else range(num_tries)
    for _ in pbar:
      order = torch.randperm(alpha_shape.numel())
      exbits = torch.zeros_like(order).to(x.device)
      exbits[order[:k]] = 1
      exbits = exbits.view(alpha_shape)
      y = model(xx, alpha=exbits, **kwargs)
      if y[0].argmax() == target_label:
        break

    if get_order:
      return exbits[0], order
    else:
      return exbits[0]


# Always gives the same alpha_star provided that X is the appropriate dimensions
class DetRandomExplainer(Explainer):
  def __init__(self, seed):
    super(DetRandomExplainer, self).__init__()
    self.seed = seed

  def __str__(self):
    return f"drand{self.seed}"

  @torch.no_grad()
  def find_explanation(self, model, x, target_label, hot_prob=0.5, get_order=False, **kwargs):
    alpha_shape = model.alpha_shape(x.unsqueeze(0))
    save_seed = torch.seed()
    torch.manual_seed(self.seed)
    order = torch.randperm(alpha_shape.numel())
    torch.manual_seed(save_seed)

    k = math.ceil(alpha_shape.numel() * hot_prob)
    exbits = torch.zeros_like(order).to(x.device)
    exbits[order[:k]] = 1
    exbits = exbits.view(alpha_shape)[0]

    if get_order:
      return exbits, order
    else:
      return exbits

# Greedy pick stuff from the pick order until the target classification is hit
@torch.no_grad()
def greedy_pick_ordering(model, x, target_label,
                         pick_order = None,
                         min_nnz_ratio = 0.2):
  model.eval()
  # If there is no pick order supplied, we just go in the order of the dimension
  x = x.unsqueeze(0)
  alpha_shape = models.alpha_shape(x)
  exbits = torch.zeros(alpha_shape).view(-1).to(x.device)

  if pick_order is None:
    pick_order = torch.tensor(range(exbits.numel()))

  assert pick_order.shape == exbits.shape
  for (i, k) in enumerate(pick_order):
    exbits[k] = 1.0
    y = model(x, exbits.view(alpha_shape))
    if y[0].argmax() == target_label and i >= min_nnz:
      break

  return exbits.view(alpha_shape)[0], pick_order

