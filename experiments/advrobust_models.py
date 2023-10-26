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

    # Run some whatevers
    base_df = q1t_test_radii(mus_base, exbits_list, imagenet_dataset, csv_saveto=base_saveto)
    l2_3_0_df = q1t_test_radii(mus_l2_3_0, exbits_list, imagenet_dataset, csv_saveto=l2_3_0_saveto)
    linf_4_df = q1t_test_radii(mus_linf_4, exbits_list, imagenet_dataset, csv_saveto=linf_4_saveto)
    linf_8_df = q1t_test_radii(mus_linf_8, exbits_list, imagenet_dataset, csv_saveto=linf_8_saveto)

    # R1 robustness checks
    check_r1_saveto = os.path.join(saveto_dir, "advmodels_check_r1.csv")
    spsz = 4
    df = pd.DataFrame(columns=[
        "mus_base", "mus_l2_3_0", "mus_linf_4", "mus_linf_8",
        "adv_base", "adv_l2_3_0", "adv_linf_4", "adv_linf_8",
    ])

    pbar = tqdm(exbits_list)
    for i, alpha in enumerate(pbar):
        x, true_label = imagenet_dataset[i]
        x, alpha = x.cuda(), alpha.cuda()
        pertb_bits = (alpha == 0).int()

        mus_base_r1_ok = check_r1_robust(mus_base, x, alpha, pertb_bits, split_size=spsz)[0]
        mus_l2_3_0_r1_ok = check_r1_robust(mus_l2_3_0, x, alpha, pertb_bits, split_size=spsz)[0]
        mus_linf_4_r1_ok = check_r1_robust(mus_linf_4, x, alpha, pertb_bits, split_size=spsz)[0]
        mus_linf_8_r1_ok = check_r1_robust(mus_linf_8, x, alpha, pertb_bits, split_size=spsz)[0]
        
        adv_base_r1_ok = check_r1_robust(adv_base, x, alpha, pertb_bits, split_size=spsz)[0]
        adv_l2_3_0_r1_ok = check_r1_robust(adv_l2_3_0, x, alpha, pertb_bits, split_size=spsz)[0]
        adv_linf_4_r1_ok = check_r1_robust(adv_linf_4, x, alpha, pertb_bits, split_size=spsz)[0]
        adv_linf_8_r1_ok = check_r1_robust(adv_linf_8, x, alpha, pertb_bits, split_size=spsz)[0]

        this_df = pd.DataFrame({
            "mus_base"   : mus_base_r1_ok,
            "mus_l2_3_0" : mus_l2_3_0_r1_ok,
            "mus_linf_4" : mus_linf_4_r1_ok,
            "mus_linf_8" : mus_linf_8_r1_ok,
            "adv_base"   : adv_base_r1_ok,
            "adv_l2_3_0" : adv_l2_3_0_r1_ok,
            "adv_linf_4" : adv_linf_4_r1_ok,
            "adv_linf_8" : adv_linf_8_r1_ok,
            }, index = [i])

        df = pd.concat([df, this_df])
        df.to_csv(emp_r1_check_saveto)

        # do the description string
        desc_str = f"mus ("
        desc_str += f"{np.array(df['mus_base']).mean():.3f}, "
        desc_str += f"{np.array(df['mus_l2_3_0']).mean():.3f}, "
        desc_str += f"{np.array(df['mus_linf_4']).mean():.3f}, "
        desc_str += f"{np.array(df['mus_linf_8']).mean():.3f}), "

        desc_str += "adv ("
        desc_str += f"{np.array(df['adv_base']).mean():.3f}, "
        desc_str += f"{np.array(df['adv_l2_3_0']).mean():.3f}, "
        desc_str += f"{np.array(df['adv_linf_4']).mean():.3f}, "
        desc_str += f"{np.array(df['adv_linf_8']).mean():.3f}), "
        
        pbar.set_description(desc_str)
        # mus (0.324, 0.432, 0.356, 0.448), adv (0.208, 0.176, 0.180, 0.232)

    df.to_csv(check_r1_saveto)


    

    

