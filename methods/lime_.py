import math
import torch
from lime import lime_image, lime_tabular

from .header import *

class LimeExplainer(Explainer):
  def __init__(self, top_k_frac, positive_inds_only=False, num_trains=100, num_samples=48, batch_size=4):
    super(LimeExplainer, self).__init__()
    self.top_k_frac = top_k_frac
    self.positive_inds_only = positive_inds_only
    self.num_trains = num_trains
    self.num_samples = num_samples
    self.batch_size = batch_size

  def __str__(self):
    return f"lime{self.top_k_frac:.4f}"

  def find_explanation(self, model, x, get_lime_exp=False, get_order=False):
    model.cuda().eval()
    xx = x.unsqueeze(0).cuda()
    y_ones = model(xx)[0]
    target_class = y_ones.argmax()
    xx_ashape = model.alpha_shape(xx)

    alpha_train_np = np.random.randint(0, 2, (self.num_trains, *xx_ashape[1:]))
    explainer = lime_tabular.LimeTabularExplainer(alpha_train_np, mode='regression')
    @torch.no_grad()
    def predict_fn(alpha_np):
      splits = torch.split(torch.tensor(alpha_np), self.batch_size)
      y = []
      for sp_alpha in splits:
        sp_x = torch.cat(sp_alpha.size(0) * [xx], dim=0)
        sp_y = model(sp_x.cuda(), alpha=sp_alpha.cuda())
        y.append(sp_y)
      y = torch.cat(y, dim=0)
      return y.detach().cpu().numpy()
    
    _, p = xx_ashape
    k = math.ceil(p * self.top_k_frac)
    alpha_test_np = np.ones(*xx_ashape[1:], dtype='long')
    lime_exp = explainer.explain_instance(alpha_test_np, predict_fn, num_samples=self.num_samples, num_features=p)
    model.to(x.device)

    # Very likely the lime_exp.local_exp[0] is already sorted, but just to make sure
    order = torch.tensor([k for (k,v) in lime_exp.local_exp[0]])
    scores = torch.tensor([v for (k,v) in lime_exp.local_exp[0]])
    order = order[scores.abs().argsort(descending=True)]

    exbits = torch.zeros(p).to(x.device)
    exbits[order[:k]] = 1.0
    if get_lime_exp:
      return exbits, lime_exp

    if get_order:
      return exbits, order

    return exbits


