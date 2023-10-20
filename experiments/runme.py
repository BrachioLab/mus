import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from experiments_header import *
from theoretical_stability import *
from empirical_stability import *
from certified_accuracy import *
from attribution_sparsities import *


# Generate all exbits, all this
def generate_all_exbits(configs, **kwargs):
    torch.manual_seed(1234)
    generate_vit16_explanations(configs)
    generate_resnet50_explanations(configs)
    generate_roberta_explanations(configs)


# Q1T
def run_theoretical_stability(configs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q1_theory")
    assert os.path.isdir(saveto_dir)
    q1t_run_stuff(configs, saveto_dir=saveto_dir)


# Q1E
# This is the most time intensive experiment, so we carefully run stuff
def run_empirical_stability(configs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q1_boxatk")
    assert os.path.isdir(saveto_dir)

	vit16_exbits_list = load_exbits_list("vit16", method_type, top_frac, configs["exbits_dir"])
	resnet50_exbits_list = load_exbits_list("resnet50", method_type, top_frac, configs["exbits_dir"])
	roberta_exbits_list = load_exbits_list("roberta", method_type, top_frac, configs["exbits_dir"])

	configs["model2exbits"] = {
		"vit16" : vit16_exbits_list,
		"resnet50" : resnet50_exbits_list,
		"roberta" : roberta_exbits_list
	}

    q1e_run_stuff("vit16", configs,
                  method_type = "shap",
                  lambds = [8/8., 4/8., 3/8., 2/8., 1/8.],
                  top_frac = 0.2500,
                  saveto_dir = saveto_dir)


# Q2
def run_certified_accuracies(configs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q2")
    assert os.path.isdir(saveto_dir)
    q2_run_stuff(configs, saveto_dir=saveto_dir)


# Q3
def run_attribution_sparsities(configs):
    torch.manual_seed(1234)
    saveto_dir = os.path.join(configs["dump_dir"], "q3_sparsity")
    assert os.path.isdir(saveto_dir)
    q3_run_stuff(configs, saveto_dir=saveto_dir)



if __name__ == "__main__":
    configs = make_default_configs()
    assert os.path.isdir(configs["dump_dir"])


