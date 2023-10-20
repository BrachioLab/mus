import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveWrapper(nn.Module):
  def __init__(self, base_model, zeta, num_samples):
    super(AdditiveWrapper, self).__init__()
    self.base_model = copy.deepcopy(base_model)
    self.zeta = zeta
    self.num_samples = num_samples

  def forward(self, x, zeta=None, num_samples=None):
    zeta = self.zeta if zeta is None else zeta
    num_samples = self.num_samples if num_samples is None else num_samples
    eps = (torch.rand(1, num_samples, *x.shape[1:]).to(x.device) - 0.5) / zeta
    xx = torch.cat(num_samples * [x.unsqueeze(1)], dim=1)
    xx_noised = xx + eps
    yy_noised = self.base_model(xx_noised.flatten(0,1))
    yy_noised = yy_noised.view(-1, num_samples, *yy_noised.shape[1:])
    yy_mean = yy_noised.mean(dim=1)
    return yy_mean


# Additive smoothing
class DumbAdditiveWrapper(nn.Module):
  def __init__(self,
               base_model,
               lambd,
               num_samples,
               use_voting = True,
               seed = 1234):
    super(DumbAdditiveWrapper, self).__init__()
    self.base_model = copy.deepcopy(base_model)
    assert lambd > 0
    self.lambd = lambd
    self.seed = seed
    self.num_samples = num_samples
    self.use_voting = use_voting

  def alpha_shape(self, x):
    raise NotImplementedError()

  def binner_product(self, x, alpha):
    raise NotImplementedError()

  def apply_additive_noise(self, x, alpha, lambd=None, num_samples=None, seed=None):
    lambd = self.lambd if lambd is None else lambd
    num_samples = self.num_samples if num_samples is None else num_samples
    
    save_seed = torch.seed()
    torch.manual_seed(self.seed if seed is None else seed)
    eps = (torch.rand(1, num_samples, *alpha.shape[1:]).to(alpha.device) - 0.5) / lambd
    torch.manual_seed(save_seed)

    alpha_noised = alpha.unsqueeze(1) + eps               # (batch_size, num_samples, *)
    xx = torch.cat(num_samples * [x.unsqueeze(1)], dim=1) # (batch_size, num_samples, *)
    xx_noised = self.binner_product(xx.flatten(0,1), alpha_noised.flatten(0,1))
    return xx_noised.view(-1, num_samples, *x.shape[1:])

  def forward(self, x, alpha=None, mu=None, num_samples=None, use_voting=None, seed=None):
    alpha = torch.ones(self.alpha_shape(x)).to(x.device) if alpha is None else alpha
    num_samples = self.num_samples if num_samples is None else num_samples
    use_voting = self.use_voting if use_voting is None else use_voting
    seed = self.seed if seed is None else seed

    xx_noised = self.apply_additive_noise(x, alpha, 
                                          lambd = self.lambd,
                                          num_samples = num_samples,
                                          seed = seed)
    yy_noised = self.base_model(xx_noised.view(-1, *x.shape[1:]))
    yy_noised = yy_noised.view(-1, num_samples, *yy_noised.shape[1:])

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


class DumbAdditiveImageNet(DumbAdditiveWrapper):
  def __init__(self, base_model, patch_size, lambd,
               num_samples = 128,
               image_shape = torch.Size([3,224,224]),
               **kwargs):
    super(AdditiveImageNet, self).__init__(base_model, lambd=lambd, num_samples=num_samples, **kwargs)
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



