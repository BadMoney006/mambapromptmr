# promptmr_ref/ README
# ===================
# This folder contains PromptMR (original model) with custom data loader.
# No Mamba modules are used here.

# Usage
# -----
# 1) Train (PromptMR)
# Edit CONFIG in `promptmr_ref/train_promptmr.py`, then run:
#   python promptmr_ref/train_promptmr.py
#
# 2) Inference/Eval (full dataset)
# Edit CONFIG in `promptmr_ref/inference_promptmr.py`, then run:
#   python promptmr_ref/inference_promptmr.py
#
# 3) One-slice visualization + metrics
# Edit CONFIG in `promptmr_ref/inference_one_slice.py`, then run:
#   python promptmr_ref/inference_one_slice.py
#
# Key parameters (train_promptmr.py)
# ---------------------------------
# root_path:       training data root (e.g. .../Cine/TrainingSet/AccFactor04)
# acc_factor:      acceleration factor string, e.g. "04"
# num_adj:         number of adjacent frames (odd, e.g. 5)
# frame_sampling:  frame sampling policy: all | center | random_one | random_k
# num_sampled_frames: K for random_k
# seed:            random seed for reproducibility
# lr:              base learning rate (2e-4)
# num_epochs:      total epochs
# batch_size:      batch size
# use_checkpoint:  memory checkpointing in PromptMR
# save_dir:        checkpoint output directory
#
# StepLR policy
# -------------
# StepLR is used with step_size = num_epochs - 1 and gamma = 0.1,
# so the learning rate drops by 10x only in the final epoch (2e-4 -> 2e-5).

# Key parameters (inference_promptmr.py)
# --------------------------------------
# root_path:       test/val root path
# model_path:      path to trained weights
# frame_sampling:  all / center / random_one / random_k
# use_crop_eval:   True to eval on center crop (H/3, W/2)
# save_dir:        optional npy output directory

# Notes
# -----
# - Dataset uses official *_mask.mat if present; otherwise falls back to nonzero mask (warns once per file).
# - Evaluation uses crop by default to align with CMRxRecon leaderboard.
# - Training uses adjacent-frame inputs; only the center frame is supervised.
