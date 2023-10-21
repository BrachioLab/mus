import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from header import *
from theoretical_stability import *
from empirical_stability import *
from certified_accuracy import *
from attribution_sparsities import *
from additive_smoothing import *
from advrobust_models import *


# Generate all exbits, all this
def generate_all_exbits(configs, num_todo=2000, **kwargs):
    torch.manual_seed(1234)
    generate_vit16_explanations(configs, num_todo=num_todo, **kwargs)
    generate_resnet50_explanations(configs, num_todo=num_todo, **kwargs)
    generate_roberta_explanations(configs, num_todo=num_todo, **kwargs)


# Q1T
def run_theoretical_stability(configs, num_todo=2000, **kwargs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q1_theory")
    assert os.path.isdir(saveto_dir)
    q1t_run_stuff(configs, num_todo=num_todo, saveto_dir=saveto_dir, **kwargs)


# Q1E: the most time intensive experiment, so be careful
def run_empirical_stability(configs, num_todo=250, **kwargs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q1_boxatk")
    assert os.path.isdir(saveto_dir)

    method_type, top_frac = "shap", 0.2500
    vit16_exbits_list = load_exbits_list("vit16", method_type, top_frac, configs["exbits_dir"])
    resnet50_exbits_list = load_exbits_list("resnet50", method_type, top_frac, configs["exbits_dir"])
    roberta_exbits_list = load_exbits_list("roberta", method_type, top_frac, configs["exbits_dir"])

    configs["model2exbits"] = {
        "vit16" : vit16_exbits_list,
        "resnet50" : resnet50_exbits_list,
        "roberta" : roberta_exbits_list
    }

    q1e_run_stuff("vit16", configs,
                  method_type = method_type,
                  lambds = [8/8., 4/8., 3/8., 2/8., 1/8.],
                  top_frac = top_frac,
                  num_todo = num_todo,
                  saveto_dir = saveto_dir,
                  **kwargs)


# Q2
def run_certified_accuracies(configs, num_todo=2000, **kwargs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q2")
    assert os.path.isdir(saveto_dir)
    q2_run_stuff(configs, num_todo=num_todo, saveto_dir=saveto_dir, **kwargs)


# Q3
def run_attribution_sparsities(configs, num_todo=250, **kwargs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q3_sparsity")
    assert os.path.isdir(saveto_dir)
    q3_run_stuff(configs, num_todo=num_todo, saveto_dir=saveto_dir, **kwargs)


# Q4
def run_additive_smoothing(configs, num_todo=2000, **kwargs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q4_additive")
    assert os.path.isdir(saveto_dir)
    run_q4_additive(configs, num_todo=num_todo, saveto_dir=saveto_dir, **kwargs)


# Q5
def run_advrobust_models(configs, num_todo=2000, **kwargs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q5_advrobust")
    assert os.path.isdir(saveto_dir)
    run_q5_advrobust(configs, num_todo=num_todo, saveto_dir=saveto_dir, **kwargs)


# Thing that actually runs
if __name__ == "__main__":
    configs = make_default_configs()
    exbits_dir, dump_dir = configs["exbits_dir"], configs["dump_dir"]

    # Make directories if they don't exist
    os.makedirs(exbits_dir, exist_ok=True)
    os.makedirs(os.path.join(dump_dir, "q1_theory"), exist_ok=True)
    os.makedirs(os.path.join(dump_dir, "q1_boxatk"), exist_ok=True)
    os.makedirs(os.path.join(dump_dir, "q2"), exist_ok=True)
    os.makedirs(os.path.join(dump_dir, "q3_sparsity"), exist_ok=True)
    os.makedirs(os.path.join(dump_dir, "q4_additive"), exist_ok=True)
    os.makedirs(os.path.join(dump_dir, "q5_advrobust"), exist_ok=True)

    if not os.path.isfile(configs["imagenet_randperm_file"]):
        torch.save(configs["imagenet_randperm"], configs["imagenet_randperm_file"])

    if not os.path.isfile(configs["tweet_randperm_file"]):
        torch.save(configs["tweet_randperm"], configs["tweet_randperm_file"])



