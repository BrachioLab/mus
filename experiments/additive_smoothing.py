import os
import copy
import torch
import torch.nn as nn

from header import *
from additive_smoothing_helpers.additive_wrappers import *
from theoretical_stability import *


def run_q4_additive(configs, num_todo=None, saveto_dir=None):
    assert saveto_dir is not None
    imagenet_dataset = configs["imagenet_dataset"]
    exbits_list = load_exbits_list("vit16", "shap", 0.25, configs["exbits_dir"], q=64, patch_size=28)
    if isinstance(num_todo, int) and num_todo > 0:
        exbits_list = exbits_list[:num_todo]

    mus_1000_saveto = os.path.join(saveto_dir, "mus_vit16_lam1000.csv")
    mus_0500_saveto = os.path.join(saveto_dir, "mus_vit16_lam0500.csv")
    mus_0250_saveto = os.path.join(saveto_dir, "mus_vit16_lam0250.csv")
    mus_0125_saveto = os.path.join(saveto_dir, "mus_vit16_lam0125.csv")
    add_1000_saveto = os.path.join(saveto_dir, "add_vit16_lam1000.csv")
    add_0500_saveto = os.path.join(saveto_dir, "add_vit16_lam0500.csv")
    add_0250_saveto = os.path.join(saveto_dir, "add_vit16_lam0250.csv")
    add_0125_saveto = os.path.join(saveto_dir, "add_vit16_lam0125.csv")

    q, psz = 64, 28
    M = q
    mus_1000 = load_model("vit16", configs["models_dir"], patch_size=psz, lambd=1.0, q=q)
    add_1000 = MuSImageNet(AdditiveWrapper(mus_1000.base_model, zeta=1.0, num_samples=M), \
                                   patch_size=psz, lambd=1.0, q=q)

    mus_0500 = load_model("vit16", configs["models_dir"], patch_size=psz, lambd=0.50, q=q)
    add_0500 = MuSImageNet(AdditiveWrapper(mus_0500.base_model, zeta=0.5, num_samples=M), \
                                   patch_size=psz, lambd=1.0, q=q)

    mus_0250 = load_model("vit16", configs["models_dir"], patch_size=psz, lambd=0.25, q=q)
    add_0250 = MuSImageNet(AdditiveWrapper(mus_0250.base_model, zeta=0.25, num_samples=M), \
                                   patch_size=psz, lambd=1.0, q=q)

    mus_0125 = load_model("vit16", configs["models_dir"], patch_size=psz, lambd=0.125, q=q)
    add_0125 = MuSImageNet(AdditiveWrapper(mus_0125.base_model, zeta=1.0, num_samples=M), \
                                   patch_size=psz, lambd=1.0, q=q)

    mus_1000.eval(), mus_0500.eval(), mus_0250.eval(), mus_0125.eval()
    add_1000.eval(), add_0500.eval(), add_0250.eval(), add_0125.eval()

    mus_1000_df = q1t_test_radii(mus_1000, exbits_list, imagenet_dataset, csv_saveto=mus_1000_saveto)
    mus_0500_df = q1t_test_radii(mus_0500, exbits_list, imagenet_dataset, csv_saveto=mus_0500_saveto)
    mus_0250_df = q1t_test_radii(mus_0250, exbits_list, imagenet_dataset, csv_saveto=mus_0250_saveto)
    mus_0125_df = q1t_test_radii(mus_0125, exbits_list, imagenet_dataset, csv_saveto=mus_0125_saveto)

    add_1000_df = q1t_test_radii(add_1000, exbits_list, imagenet_dataset, csv_saveto=add_1000_saveto)
    add_0500_df = q1t_test_radii(add_0500, exbits_list, imagenet_dataset, csv_saveto=add_0500_saveto)
    add_0250_df = q1t_test_radii(add_0250, exbits_list, imagenet_dataset, csv_saveto=add_0250_saveto)
    add_0125_df = q1t_test_radii(add_0125, exbits_list, imagenet_dataset, csv_saveto=add_0125_saveto)

