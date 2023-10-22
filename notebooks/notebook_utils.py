import os
import pathlib
import pandas as pd
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DUMP_DIR = os.path.join(BASE_DIR, "dump")

Q1_THEORY_DIR = os.path.join(DUMP_DIR, "q1_theory")
Q1_BOXATK_DIR = os.path.join(DUMP_DIR, "q1_boxatk")
Q2_CERTACC_DIR = os.path.join(DUMP_DIR, "q2_certacc")
Q3_SPARSITY_DIR = os.path.join(DUMP_DIR, "q3_sparsity")
Q4_ADDITIVE_DIR = os.path.join(DUMP_DIR, "q4_additive")
Q5_ADVROBUST_DIR = os.path.join(DUMP_DIR, "q4_advrobust")

""" Q1 Stuff
"""
def load_q1_theory(model_type, method_type, top_frac, lambd):
    model_type = model_type.lower()
    method_type = method_type.lower()
    method_type = "vgradu" if method_type == "vgrad" else method_type
    method_type = "igradu" if method_type == "igrad" else method_type
    
    if model_type == "roberta":
        csv_file = f"q1t_{model_type}_q64_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
    else:
        csv_file = f"q1t_{model_type}_psz28_q64_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
    csv_file = os.path.join(Q1_THEORY_DIR, csv_file)
    return pd.read_csv(csv_file)

def load_q1_boxatk(model_type, method_type, top_frac, lambd):
    if model_type == "roberta":
        csv_file = f"q1e_{model_type}_q16_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
    else:
        csv_file = f"q1e_{model_type}_psz28_q16_{method_type}_top{top_frac:.4f}_lam{lambd:.4f}.csv"
    csv_file = os.path.join(Q1_BOXATK_DIR, csv_file)
    return pd.read_csv(csv_file)

# Certified stuff
def q1t_drops(df, rs, prop="inc", consistent=True, div_by_p=False, use_mu=False):
    assert prop in ["inc", "dec"]
    
    true_labels = df["true_label"].to_numpy()
    ones_labels = (df["ones_mu_label"] if use_mu else df["ones_label"]).to_numpy()
    exbs_labels = (df["exbs_mu_label"] if use_mu else df["exbs_label"]).to_numpy()
    ones_gaps = (df["ones_mu_gap"] if use_mu else df["ones_gap"]).to_numpy()
    exbs_gaps = (df["exbs_mu_gap"] if use_mu else df["exbs_gap"]).to_numpy()
    todo_labels = exbs_labels if prop == "inc" else ones_labels
    todo_gaps = exbs_gaps if prop == "inc" else ones_gaps
    
    N = len(df)
    lambd = df["lambd"][0]
    cert_rs = todo_gaps / (2 * lambd)
    
    if div_by_p:
        ps = df["p"].to_numpy()
        cert_rs = cert_rs / ps
    
    if consistent:
        hit_bits = ones_labels == exbs_labels
        drops = np.array([hit_bits[cert_rs >= r].sum() / N for r in rs])
    else:
        drops = np.array([(cert_rs >= r).sum() / N for r in rs])

    return drops

# Box attack stuff
def q1e_drops(df, rs, prop="inc", consistent=True, div_by_p=False, use_mu=False):
    assert prop in ["inc", "dec"]
    
    true_labels = df["true_label"].to_numpy()
    ones_labels = (df["ones_mu_label"] if use_mu else df["ones_label"]).to_numpy()
    exbs_labels = (df["exbs_mu_label"] if use_mu else df["exbs_label"]).to_numpy()
    todo_labels = exbs_labels if prop == "inc" else ones_labels
    r_maxs = (df["inc_curr_r_max"] if prop == "inc" else df["dec_curr_r_max"]).to_numpy()
        
    N = len(df)
    r_maxs = r_maxs / df["p"].to_numpy() if div_by_p else r_maxs
    if consistent:
        hit_bits = ones_labels == exbs_labels
        drops = np.array([hit_bits[r_maxs >= r].sum() / N for r in rs])
    else:
        drops = np.array([(r_maxs >= r).sum() / N for r in rs])
    return drops



""" Q2 Stuff
"""
def load_q2(model_type, q, lambd):
    if model_type == "roberta":
        csv_file = f"q2_{model_type}_q{q}_lam{lambd:.4f}.csv"
    else:
        csv_file = f"q2_{model_type}_psz28_q{q}_lam{lambd:.4f}.csv"
    csv_file = os.path.join(Q2_CERTACC_DIR, csv_file)
    return pd.read_csv(csv_file)

def q2_drops(df, rs, div_by_p=False):
    true_labels = df["true_label"].to_numpy()
    ones_labels = df["ones_label"].to_numpy()
    N = len(df)
    hit_bits = true_labels == ones_labels
    cert_rs = df["cert_r"].to_numpy()
    if div_by_p:
        ps = df["p"].to_numpy()
        cert_rs = cert_rs / ps
    drops = np.array([hit_bits[cert_rs >= r].sum() / N for r in rs])
    return drops


""" Q3 Stuff
"""
def load_q3_df(model_type, method_type, lambd):
    if model_type == "roberta":
        csv_file = f"q3s_{model_type}_q64_{method_type}_lam{lambd:.4f}.csv"
    else:
        csv_file = f"q3s_{model_type}_psz28_q64_{method_type}_lam{lambd:.4f}.csv"
    csv_file = os.path.join(Q3_SPARSITY_DIR, csv_file)
    return pd.read_csv(csv_file)

def q3_fracs(df):
    p = df["p"].to_numpy()
    k = df["k"].to_numpy()
    return (k/p).mean()

def q3_accs(df):
    N = len(df)
    true_labels = df["true_label"].to_numpy()
    exbs_labels = df["exbs_label"].to_numpy()
    return (true_labels == exbs_labels).sum() / N

