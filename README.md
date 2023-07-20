# Multiplicative Smoothing

# Experiment Setup
* Download TweetEval into this directory: https://github.com/cardiffnlp/tweeteval
* Make sure you have ImageNet1K somewhere
* Modify paths in `qheader.py` as needed
* `mkdir -p dump/q1_theory dump/q1_boxatk dump/q2 dump/q3_sparsity`
* `mkdir -p notebooks/images`
* `mkdir -p saved_models`

# Training
To train Vision Transformer with `lambd = 0.6` and `lambd = 0.3` masking for 2 epochs,
* `python3 -i fine_tuning/imagenet_fine_tuning -m vit16 -d /path/to/imagenet/root -s saved_models/ -l 0.6 0.3 -e 2 --go`
To train Vision Transformer with `lambd = 0.6` and `lambd = 0.3` masking for 2 epochs,
* `python3 -i fine_tuning/imagenet_fine_tuning -m resnet50 -d /path/to/imagenet/root -s saved_models/ -l 0.6 0.3 -e 2 --go`
To train TweetEval classifiers with `lambd = 0.6` and `lambd = 0.3` masking for 2 epochs, do:
* `python3 -i fine_tuning/tweet_fine_tuning.py -s saved_models/ -l 0.6 0.3 -e 2 --go`


# Experiments Scripts

The following scripts will put data into the `dump/` directory. Some of these experiments take a long time, and the data in `dump/` have been pre-generated.

* E1: Code is in `q1_theory.py` and `q1_boxatk.py`.
  - `python3 -i q1_theory.py` and run:
    + `q1t_run_stuff(configs)`
  - `python3 -i q1_boxatk.py` and run:
    + `q1e_run_stuff("vit16", configs)`
    + `q1e_run_stuff("resnet50", configs)`
    + `q1e_run_stuff("roberta", configs)`
* E2: Code is in `q2.py`
  - `python3 -i q2.py` and run:
    + `q2_run_stuff(configs)`
* E3: Code is in `q3_sparsity.py`
  - `python3 -i q3_sparsity` and run:
    + `q3_run_stuff(configs)`
    
# Generate Plots
* Load up `notebooks/neurips_plots.ipynb` and run the entire file.
