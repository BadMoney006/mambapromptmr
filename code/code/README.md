# code/ README
# =================
# This folder contains MambaPromptMR training/inference scripts and dataset.

# Usage
# -----
# 1) Train (MambaPromptMR)
# Edit CONFIG in `code/train.py`, then run:
#   python code/train.py
#
# 2) Full inference/eval (all slices/frames)
# Edit CONFIG in `code/mamba_prompt_unet/inference_full.py`, then run:
#   python code/mamba_prompt_unet/inference_full.py
#
# 3) Debug one sample (single case)
# Edit INFER_CONFIG in `code/mamba_prompt_unet/inference_debug.py`, then run:
#   python code/mamba_prompt_unet/inference_debug.py
#
# 4) One-slice visualization + metrics
# Edit CONFIG in `code/mamba_prompt_unet/inference_one_slice.py`, then run:
#   python code/mamba_prompt_unet/inference_one_slice.py
#
# Key parameters (train.py)
# -------------------------
# root_path:       training data root (e.g. .../Cine/TrainingSet/AccFactor04)
# acc_factor:      acceleration factor string, e.g. "04"
# num_adj:         number of adjacent frames (odd, e.g. 5)
# frame_sampling:  frame sampling policy: all | center | random_one | random_k
# num_sampled_frames: K for random_k
# seed:            random seed for reproducibility
# lr:              base learning rate
# min_lr:          cosine annealing minimum lr
# num_epochs:      total epochs
# batch_size:      batch size
# use_amp:         mixed precision on/off
# save_dir:        checkpoint output directory
#
# Key parameters (inference_full.py)
# ----------------------------------
# val_root:        test/val root path
# model_path:      path to trained weights
# frame_sampling:  all / center / random_one / random_k
# num_sampled_frames: K for random_k
# use_crop_eval:   True to eval on center crop (H/3, W/2)
# save_dir:        output images directory
#
# Notes
# -----
# - Dataset uses official *_mask.mat if present; otherwise falls back to nonzero mask (warns once per file).
# - Evaluation uses crop by default to align with CMRxRecon leaderboard.
# - Training uses adjacent-frame inputs; only the center frame is supervised.
