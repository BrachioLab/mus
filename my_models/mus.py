import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper around the classification model
class MuS(nn.Module):
  def __init__(self, base_model, # Batched classification model
                     q,          # Quantization amount
                     lambd,      # The lambda to use
                     use_voting = True, # True: 0/1 voting, False: average logits
                     return_all_q = False,
                     seed = 1234):      # RNG seed
    super(MuS, self).__init__()
    self.base_model = copy.deepcopy(base_model)
    self.q = q
    self.lambd = max(1/q, int(lambd*q)/q)
    self.use_voting = use_voting
    self.return_all_q = return_all_q
    self.seed = seed

  # The shape of the noise
  def alpha_shape(self, x):
    raise NotImplementedError()

  # How to actually combine x: (M,*) and alpha: (M,*), typically M = N*q
  def binner_product(self, x, alpha):
    raise NotImplementedError()

  # Apply the noise
  def apply_mus_noise(self, x, alpha, mu, v=None, seed=None):
    alflat, muflat, q = alpha.flatten(1), mu.flatten(1), self.q
    N, p = alflat.shape
    if v is None:
      save_seed = torch.seed()
      torch.manual_seed(self.seed if seed is None else seed)
      v = (torch.randint(0, q, (p,)) / q).to(x.device)
      torch.manual_seed(save_seed)

    # s_base has q total values from {0, 1/q, ..., (q-1)/q} + 1/(2q)
    s_base = ((torch.tensor(range(0,q)) + 0.5) / q).to(x.device)
    t = (v.view(1,p) + s_base.view(q,1)).remainder(1.0) # (q,p)
    # Equivalent to: s = (t <= self.lambd).float() # (q,p)
    s = (2 * self.q * F.relu(self.lambd - t)).clamp(0,1) # (q,p)

    talpha = (muflat.view(N,1,p) + (alflat.view(N,1,p) * s.view(1,q,p))).clamp(0,1)
    talpha = talpha.view(N*q,*alpha.shape[1:]) # (Nq, *)

    xx = torch.cat(q * [x.unsqueeze(1)], dim=1).flatten(0,1) # (Nq, *)
    xx_noised = self.binner_product(xx, talpha)
    return xx_noised.view(N,q,*x.shape[1:])

  # Forward
  def forward(self, x,
              alpha = None,     # Binary vector (N,p), defaults to ones
              mu = None,        # Binary vector (N,p), defaults to zeros
              eps = 1e-5,
              return_all_q = None,
              use_voting = None,
              **kwargs):
    alpha = torch.ones(self.alpha_shape(x)).to(x.device) if alpha is None else alpha
    mu = torch.zeros_like(alpha).to(x.device) if mu is None else mu
    return_all_q = self.return_all_q if return_all_q is None else return_all_q
    use_voting = self.use_voting if use_voting is None else use_voting
    assert self.alpha_shape(x) == alpha.shape and alpha.shape == mu.shape
    assert (mu - alpha).max() < eps # mu <= alpha

    # If we're small, just skip
    if abs(self.lambd - 1.0) < 0.5 / self.q:
      xx_noised = self.binner_product(x, torch.maximum(mu, alpha))
      yy_noised = self.base_model(xx_noised, **kwargs)
      yy_noised = yy_noised.view(-1,1,*yy_noised.shape[1:])
    else:
      xx_noised = self.apply_mus_noise(x, alpha, mu, seed=self.seed)  # (N,q,*)
      yy_noised = self.base_model(xx_noised.flatten(0,1), **kwargs)
      yy_noised = yy_noised.view(-1,self.q,*yy_noised.shape[1:])

    # If we want to return everything, do so
    if return_all_q:
      return yy_noised

    # ... Otherwise (for simplicity) we need to assume that last dim is the classes
    assert yy_noised.ndim == 3
    if use_voting:
      yy_noised = F.one_hot(yy_noised.argmax(dim=2), yy_noised.shape[-1]).float()
    else:
      # If not normalized, do so
      yy_sum = yy_noised.sum(dim=1)
      if (yy_noised.min() < -eps
          or yy_noised.max() > 1 + eps
          or yy_sum.max() > 1 + eps
          or yy_sum.min() < 1 - eps):
        yy_noised = yy_noised.softmax(dim=2)

    return yy_noised.mean(dim=1)

# Simple guy for simple needs
class SimpleMuS(MuS):
  def __init__(self, base_model, q=64, lambd=16/64):
    super(SimpleMuS, self).__init__(base_model, q=q, lambd=lambd)

  def alpha_shape(self, x):
    return x.shape

  def binner_product(self, x, alpha):
    assert x.shape == alpha.shape
    return x * alpha

