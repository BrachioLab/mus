import torch
import torch.nn as nn
import math
import itertools
from tqdm import tqdm
from my_models import *

# The default split size to use

# Does the particular alpha preserve the prediction of using the whole X?
@torch.no_grad()
def check_consistent(model, x, alpha):
  ref_label = model(x.unsqueeze(0))[0].argmax()
  alpha_label = model(x.unsqueeze(0), alpha=alpha.unsqueeze(0))[0].argmax()
  return ref_label == alpha_label

# Check certified incremental / decremental stability radius
@torch.no_grad()
def check_cert_r(model, x, alpha):
  xx = x.unsqueeze(0)
  aa = alpha.unsqueeze(0)
  onesp = torch.ones_like(aa)
  zerosp = torch.zeros_like(aa)
  x_test = torch.cat([xx, xx, xx, xx], dim=0)
  alpha_test = torch.cat([aa, onesp, aa, onesp], dim=0)
  mu_test = torch.cat([zerosp, zerosp, aa, aa], dim=0)

  y = model(x_test, alpha=alpha_test, mu=mu_test)
  logits, order = y.sort(dim=1, descending=True)
  pA, pB = logits[:,0], logits[:,1]
  cert_r = (pA - pB) / (2 * model.lambd)
  inc_r, dec_r, inc_mu_r, dec_mu_r = cert_r
  return inc_r.item(), dec_r.item(), inc_mu_r.item(), dec_mu_r.item()

# N samples of r elements from a tensor x: n
# returns: (N, r)
def sample_elems(x, r, N):
  assert x.ndim == 1
  n = x.shape[0]
  samples = [x[torch.randperm(n)[:r]].view(1,-1) for i in range(N)]
  return torch.cat(samples, dim=0)

# Check multiple perturbations at some radius, sampled from pert_inds
# Exception at radius 1, where we check everything
@torch.no_grad()
def check_r1_robust(model, x, alpha, pertb_bits, split_size=24):
  assert pertb_bits.ndim == 1
  pertb_inds = pertb_bits.nonzero().view(-1)
  alpha_pertbs = alpha.view(1,-1).repeat(pertb_inds.size(0),1)
  todo_inds = pertb_inds.view(-1,1)
  for i, tinds in enumerate(todo_inds):
    alpha_pertbs[i,tinds] = 1 - alpha_pertbs[i,tinds]
  pertb_labels = []
  splits = torch.split(alpha_pertbs, split_size)
  sp_x = torch.cat(split_size * [x.unsqueeze(0)], dim=0)
  for sp_alpha in splits:
    if sp_alpha.size(0) < split_size:
      sp_x = torch.cat(sp_alpha.size(0) * [x.unsqueeze(0)], dim=0)
    sp_labels = model(sp_x, alpha=sp_alpha).argmax(dim=1)
    pertb_labels.append(sp_labels)
  pertb_labels = torch.cat(pertb_labels, dim=0)
  # There exists a counter example, so the property fails
  target_label = model(x.unsqueeze(0), alpha=alpha.unsqueeze(0))[0].argmax()
  if (target_label != pertb_labels).sum() > 0:
    bad_ind = (pertb_labels != target_label).nonzero()[0]
    bad_alpha = alpha_pertbs[bad_ind]
    return False, bad_alpha
  # Good to return
  else:
    return True, None

# Sample N stuff that are <= r_max wrt x
def sample_some_elems(x, r_max, N):
  assert x.ndim == 1
  n = x.shape[0]
  rs = torch.randint(1, r_max+1, (N,))
  samples = [x[torch.randperm(n)[:r]] for r in rs]
  return samples

# The empirical radius via box attack
@torch.no_grad()
def find_emp_stability(model, x, hots, pertb_bits, max_iters,
                       init_r_max = None,
                       local_r_max = None,
                       num_pertbs = 24,  # Or whatever fits on your GPU
                       max_resets = None,
                       do_r1_check = True,
                       ce_iter_giveup_gap = None,
                       progress_bar = False):
  assert isinstance(model, MuS)
  # Set up some local search parameters
  pertb_inds = pertb_bits.nonzero().view(-1)
  r_max = pertb_inds.numel() if init_r_max is None else min(init_r_max, pertb_inds.numel())
  local_r_max = max(r_max//5, 3) if local_r_max is None else min(local_r_max, r_max)
  max_resets = max_iters if max_resets is None else max_resets
  ce_iter_giveup_gap = max_iters if ce_iter_giveup_gap is None else ce_iter_giveup_gap

  # Basic sanity checks on whether to continue
  if pertb_inds.numel() == 0 or r_max <= 0:
    print("find_emp_stability: trivial with pertb_inds.numel = {pertb_inds.numel()}, r_max {r_max}")
    return None

  init_y = model(x.unsqueeze(0), alpha=hots.unsqueeze(0))[0]
  target_label = init_y.argmax(0)
  loss_fn = nn.CrossEntropyLoss(reduction="none") # reduction="none" means vector of loss values

  # Need to maintain the invariant that ||hots - curr_hots||_0 <= r_max
  x_pertbs = torch.cat(num_pertbs * [x.unsqueeze(0)], dim=0)
  init_loss = loss_fn(init_y, target_label)
  curr_hots, curr_loss = hots.clone(), init_loss

  # The new radius we can certify so far
  num_resets = 0
  num_locr0_resets = 0
  flips_by_loss = 0
  flips_by_rand = 0
  max_iter_dist = 0
  prev_ce_iter = 0 # Haven"t found anything yet
  num_ces = 0
  best_ce = None
  skip_loop = False

  if do_r1_check or r_max == 1:
    all_good, ce = check_r1_robust(model, x, hots, pertb_bits)
    # If we fail the check, set some flags so that we immediately skip the loop
    if not all_good:
      r_max = 0 # Set this so we have a reliable curr_r_max tracking
      best_ce = ce
      num_ces += 1
      skip_loop = True

  max_iters = tqdm(range(max_iters)) if progress_bar else range(max_iters)
  for _iter in max_iters:
    if skip_loop: break
    if r_max <= 1: break
    if num_resets > max_resets: break
    if _iter - prev_ce_iter > ce_iter_giveup_gap: break

    curr_dist = (hots - curr_hots).abs().sum().int().item()
    max_iter_dist = max(max_iter_dist, curr_dist)
    assert curr_dist <= r_max
    local_r = min(r_max-curr_dist, local_r_max)

    # Reset immediately if local radius is zero, since since no possible pertbs
    if local_r == 0:
      curr_hots, curr_loss = hots.clone(), init_loss
      num_resets += 1
      num_locr0_resets += 1
      continue

    # Sample a bunch of perturbation to try
    todo_inds = sample_some_elems(pertb_inds, local_r, num_pertbs)
    hots_pertbs = curr_hots.view(1,-1).repeat(num_pertbs,1)
    for i, tinds in enumerate(todo_inds):
      hots_pertbs[i,tinds] = 1 - hots_pertbs[i,tinds]

    # The labels; enable averaging logits so we can have a more sensitive loss value
    y_pertbs = model(x_pertbs, alpha=hots_pertbs)
    pertb_labels = y_pertbs.argmax(dim=1)
    pertb_losses = loss_fn(y_pertbs, target_label.repeat(y_pertbs.size(0)))

    # Did we get any decision flips? If so immediately constrain our r_max and reset things
    if (pertb_labels != target_label).sum() > 0:
      bad_ind = (pertb_labels != target_label).nonzero()[0]
      bad_hots = hots_pertbs[bad_ind]
      r_max = (hots - bad_hots).abs().sum().int().item() - 1
      curr_hots, curr_loss = hots.clone(), init_loss
      # print(f"\ncounter example hit at radius {r_max+1}\n")
      num_resets += 1
      num_ces += 1
      prev_ce_iter = _iter
      best_ce = bad_hots

    # Did anything surpass the prev loss? If so, update the curr_hots
    elif pertb_losses.max() > curr_loss:
      # Update curr_hots with the highest ind
      curr_hots, curr_loss = hots_pertbs[pertb_losses.argmax()].clone(), pertb_losses.max()
      flips_by_loss += 1

    # If we"ve been at this for a while: pick a new index such that:
    #   hots[ind] == curr_hots[ind] and inds in pertb_inds
    else:
      num_to_flip = torch.randint(1, r_max, ()).item()
      flip_inds = sample_elems(pertb_inds, num_to_flip, 1)[0]
      curr_hots = hots.clone()
      curr_hots[flip_inds] = 1 - curr_hots[flip_inds]
      curr_y = model(x.unsqueeze(0), alpha=curr_hots.unsqueeze(0))[0]
      curr_loss = loss_fn(curr_y, target_label)
      num_resets += 1
      flips_by_rand += 1

    if progress_bar:
      desc_str = f"rmax {r_max} ({init_r_max}), "
      desc_str += f"nces {num_ces}, "
      desc_str += f"rst ({num_resets}, z {num_locr0_resets}), "
      desc_str += f"fbl {flips_by_loss}, "
      desc_str += f"fbr {flips_by_rand}, "
      desc_str += f"crd {curr_dist}, "
      desc_str += f"lcr {local_r}, "
      max_iters.set_description(desc_str)

  if (not do_r1_check) and r_max == 1:
    all_good, ce = check_r1_robust(model, x, hots, pertb_bits)
    r_max = 1 if all_good else 0
    num_ces += 0 if all_good else 1
    best_ce = best_ce if all_good else ce
  best_ce = None if best_ce is None else best_ce.detach().cpu()

  stats = { "total_iters" : _iter,
            "init_r_max" : init_r_max,
            "curr_r_max" : r_max,
            "num_resets" : num_resets,
            "num_locr0_resets" : num_locr0_resets,
            "flips_by_loss" : flips_by_loss,
            "flips_by_rand" : flips_by_rand,
            "max_iter_dist" : max_iter_dist,
            "prev_ce_iter" : prev_ce_iter,
            "num_ces" : num_ces,
            "best_ce" : best_ce
          }
  return stats

#
@torch.no_grad()
def find_emp_inc_stability(model, x, hots, todo_init_r_maxs,
                           max_iters = -1,
                           max_resets = None,
                           do_r1_check = True,
                           progress_bar = False):
  assert max_iters > 0
  pertb_bits = (hots == 0).int()
  if pertb_bits.sum() == 0: return 0
  for init_r_max in todo_init_r_maxs:
    voting_save = model.use_voting
    model.use_voting = False
    stats = find_emp_stability(model, x, hots, pertb_bits,
                               init_r_max = init_r_max,
                               max_iters = max_iters,
                               max_resets = max_resets,
                               do_r1_check = do_r1_check,
                               progress_bar = progress_bar)
    model.use_voting = voting_save
    # Return early if possible
    if stats is None: return None
    if stats["num_ces"] > 0: return stats
  # Otherwise return the last loop's
  return stats

#
@torch.no_grad()
def find_emp_dec_stability(model, x, hots, todo_init_r_maxs,
                           max_iters = -1,
                           max_resets = None,
                           do_r1_check = True,
                           progress_bar = False):
  assert max_iters > 0
  pertb_bits = (hots == 0).int()
  if pertb_bits.sum() == 0: return 0
  beta = torch.ones_like(hots).to(hots.device)
  for init_r_max in todo_init_r_maxs:
    voting_save = model.use_voting
    model.use_voting = False
    stats = find_emp_stability(model, x, beta, pertb_bits,
                               init_r_max = init_r_max,
                               max_iters = max_iters,
                               max_resets = max_resets,
                               do_r1_check = do_r1_check,
                               progress_bar = progress_bar)
    model.use_voting = voting_save
    # Return early if possible
    if stats is None: return None
    if stats["num_ces"] > 0: return stats
  # Otherwise return the last loop's
  return stats

