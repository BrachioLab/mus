import os
import sys
import copy
import pathlib

sys.path.insert(0, os.path.join(str(pathlib.Path(__file__).parent.resolve()),
                                "advrobust_models_helpers", "robustness"))

from my_models import *
from theoretical_stability import *
from header import *
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

def run_q5_advrobust(configs, num_todo=None, saveto_dir=None):
    assert saveto_dir is not None
    imagenet_dataset = configs["imagenet_dataset"]
    exbits_list = load_exbits_list("resnet50", "shap", 0.25, configs["exbits_dir"], q=64, patch_size=28)
    if isinstance(num_todo, int) and num_todo > 0:
        exbits_list = exbits_list[:num_todo]

    l2_3_0_path = os.path.join(configs["models_dir"], "imagenet_l2_3_0.pt")
    linf_4_path = os.path.join(configs["models_dir"], "imagenet_linf_4.pt")
    linf_8_path = os.path.join(configs["models_dir"], "imagenet_linf_8.pt")
    assert os.path.isfile(l2_3_0_path)
    assert os.path.isfile(linf_4_path)
    assert os.path.isfile(linf_8_path)

    base_saveto = os.path.join(saveto_dir, "resnet50_base.csv")
    l2_3_0_saveto = os.path.join(saveto_dir, "WjBG_resnet50_l2_3_0.csv")
    linf_4_saveto = os.path.join(saveto_dir, "WjBG_resnet50_linf_4.csv")
    linf_8_saveto = os.path.join(saveto_dir, "WjBG_resnet50_linf_8.csv")

    raw_adv_base = pt_models.resnet50(weights=pt_models.ResNet50_Weights.IMAGENET1K_V1)
    raw_adv_base.cpu()

    ids = ImageNet(configs["imagenet_val_dir"])
    raw_adv_l2_3_0, _ = make_and_restore_model(arch="resnet50", dataset=ids, resume_path=l2_3_0_path)
    raw_adv_linf_4, _ = make_and_restore_model(arch="resnet50", dataset=ids, resume_path=linf_4_path)
    raw_adv_linf_8, _ = make_and_restore_model(arch="resnet50", dataset=ids, resume_path=linf_8_path)

    raw_adv_l2_3_0 = raw_adv_l2_3_0.model.cpu()
    raw_adv_linf_4 = raw_adv_linf_4.model.cpu()
    raw_adv_linf_8 = raw_adv_linf_8.model.cpu()

    mus_base = MuSImageNet(MyResNet(raw_adv_base), patch_size=28, q=64, lambd=0.25).eval()
    mus_l2_3_0 = MuSImageNet(MyResNet(raw_adv_l2_3_0), patch_size=28, q=64, lambd=0.25).eval()
    mus_linf_4 = MuSImageNet(MyResNet(raw_adv_linf_4), patch_size=28, q=64, lambd=0.25).eval()
    mus_linf_8 = MuSImageNet(MyResNet(raw_adv_linf_8), patch_size=28, q=64, lambd=0.25).eval()

    adv_base = MuSImageNet(MyResNet(raw_adv_base), patch_size=28, q=64, lambd=1.0).eval()
    adv_l2_3_0 = MuSImageNet(MyResNet(raw_adv_l2_3_0), patch_size=28, q=64, lambd=1.0).eval()
    adv_linf_4 = MuSImageNet(MyResNet(raw_adv_linf_4), patch_size=28, q=64, lambd=1.0).eval()
    adv_linf_8 = MuSImageNet(MyResNet(raw_adv_linf_8), patch_size=28, q=64, lambd=1.0).eval()

    base_df = q1t_test_radii(mus_base, exbits_list, imagenet_dataset, csv_saveto=base_saveto)
    l2_3_0_df = q1t_test_radii(mus_l2_3_0, exbits_list, imagenet_dataset, csv_saveto=l2_3_0_saveto)
    linf_4_df = q1t_test_radii(mus_linf_4, exbits_list, imagenet_dataset, csv_saveto=linf_4_saveto)
    linf_8_df = q1t_test_radii(mus_linf_8, exbits_list, imagenet_dataset, csv_saveto=linf_8_saveto)


