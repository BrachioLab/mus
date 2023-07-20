import math
import torch
import torch.nn as nn
import shap

import copy

from .header import *

# Wrap something around a MuS model so that we can exploit CUDA better
class ShapCudaMuSWrapper(nn.Module):
  def __init__(self, model, x):
    super(ShapCudaMuSWrapper, self).__init__()
    self.model = model.cuda()
    assert isinstance(x, torch.Tensor)
    self.x = x.cuda()
  
  def forward(self, alpha):
    torch.cuda.empty_cache()
    N, p = alpha.shape
    device = alpha.device
    xx = torch.stack(N*[self.x])
    y = self.model(xx, alpha=alpha.cuda())
    return y.to(device)

#
class GradShapExplainer(Explainer):
  def __init__(self, top_k_frac, num_trains=100, num_samples=48, shap_batch_size=4):
    super(GradShapExplainer, self).__init__()
    self.top_k_frac = top_k_frac
    self.num_trains = num_trains
    self.num_samples = num_samples
    self.shap_batch_size = shap_batch_size

  def __str__(self):
    return f"shap{self.top_k_frac:.4f}"

  def find_explanation(self, model, x, get_shap_values=False, get_order=False, **kwargs):
    model.eval()

    xx = x.unsqueeze(0)
    xx_ashape = model.alpha_shape(xx)
    alpha_train = torch.randint(0,2,(self.num_trains,*xx_ashape[1:]))
    alpha_train[0] = torch.ones(*xx_ashape[1:])

    cuda_model = ShapCudaMuSWrapper(model, x)
    explainer = shap.GradientExplainer(cuda_model, [alpha_train], batch_size=self.shap_batch_size)
    raw_shap_values, _ = explainer.shap_values([torch.ones(xx_ashape)], ranked_outputs=1, nsamples=self.num_samples)
    shap_values = torch.tensor(raw_shap_values[0][0])

    # The explanation
    k = math.ceil(shap_values.numel() * self.top_k_frac)
    order = shap_values.argsort(descending=True)
    exbits = torch.zeros(xx_ashape[1:]).to(x.device)
    exbits[order[:k]] = 1.0
    
    if get_shap_values:
      return exbits, raw_shap_values

    if get_order:
      return exbits, order

    return exbits


