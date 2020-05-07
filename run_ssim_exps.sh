#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)_ssim

###Baseline (SS-DPC) + ssim
python3 run_mono_dpc.py --date May03_ssim --train_seq 'all' --val_seq '00' --test_seq '09' --use_ssim