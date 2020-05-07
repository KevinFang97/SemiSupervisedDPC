#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)_smo

###Baseline (SS-DPC) + ssim + smo
python3 run_mono_dpc.py --date May03_smo --train_seq 'all' --val_seq '00' --test_seq '09' --use_ssim --smo_weight 0.002