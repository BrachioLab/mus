import torch
import math
from .header import *

# Vanilla gradient saliency
class PlainGradExplainer(Explainer):
  def __init__(self, top_k_frac, signed=False):
    super(PlainGradExplainer, self).__init__()
    self.top_k_frac = top_k_frac
    self.signed = signed

  def __str__(self):
    if self.signed:
      return f"pgrads{self.top_k_frac:.4f}"
    else:
      return f"pgradu{self.top_k_frac:.4f}"

  def _find_grad_order(self, model, x):
    model.eval()
    xx = x.unsqueeze(0)
    test = torch.ones(model.alpha_shape(xx))[0].to(xx.device)
    test.requires_grad_()
    y = model(xx, alpha=test.unsqueeze(0))[0]
    v = y.max()
    v.backward(retain_graph=True)
    grad = test.grad.view(-1)
    order = grad.argsort(descending=True) if self.signed else grad.abs().argsort(descending=True)
    return grad.view(test.shape), order.view(test.shape)

  def find_explanation(self, model, x, get_order=False):
    model.eval()
    _, order = self._find_grad_order(model, x)
    
    alpha_shape = model.alpha_shape(x.unsqueeze(0))[1:]
    k = math.ceil(order.numel() * self.top_k_frac)
    exbits = torch.zeros(order.numel()).to(x.device)
    exbits[order.view(-1)[:k]] = 1.0
    if get_order:
      return exbits.view(*alpha_shape), order
    else:
      return exbits.view(*alpha_shape)

# Integrated gradient saliency
class IntGradExplainer(Explainer):
  def __init__(self, top_k_frac, signed=False, num_steps=32):
    super(IntGradExplainer, self).__init__()
    self.top_k_frac = top_k_frac
    self.signed = signed
    self.num_steps = num_steps

  def __str__(self):
    if self.signed:
      return f"igrads{self.top_k_frac:.4f}"
    else:
      return f"igradu{self.top_k_frac:.4f}"

  def _intgrad(self, model, x, num_steps=None):
    num_steps = self.num_steps if num_steps is None else num_steps
    xx = x.unsqueeze(0)
    y = model(xx)
    target_class = y.argmax(dim=1)[0]
    alpha_shape = model.alpha_shape(xx)

    intgrad = torch.zeros(alpha_shape).to(x.device)
    exbits_start = torch.zeros(alpha_shape).to(x.device)
    exbits_final = torch.ones(alpha_shape).to(x.device)

    for k in range(num_steps):
      exbits_this = exbits_start + (k/num_steps) * exbits_final
      exbits_this.requires_grad_()
      y_this = model(xx, alpha=exbits_this)
      y_target = y_this[0, target_class]
      y_target.backward()
      intgrad += (exbits_this.grad / num_steps) * (exbits_final - exbits_start)
    return intgrad[0]
  
  def find_explanation(self, model, x, get_order=False):
    model.eval()
    intgrad = self._intgrad(model, x)

    alpha_shape = model.alpha_shape(x.unsqueeze(0))[1:]
    tmp = intgrad.view(-1)
    order = tmp.argsort(descending=True) if self.signed else tmp.abs().argsort(descending=True)
    k = math.ceil(order.numel() * self.top_k_frac)
    exbits = torch.zeros(order.numel()).to(x.device)
    exbits[order.view(-1)[:k]] = 1.0
    if get_order:
      return exbits.view(*alpha_shape), order
    else:
      return exbits.view(*alpha_shape)


