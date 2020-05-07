#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)_ssim

###Baseline (SS-DPC) + ssim
python3 run_mono_dpc.py --date May05_ssim_su --val_seq '00' --test_seq '09' --num_epochs 9 --lr 3e-5 --wd 3e-6 --lr_decay_epoch 3 --supervised --model_dir /home/mscv/npc/SemiSupervisedDPC/results/offline-mode/May03_ssim/2020-5-3-0-45-val_seq-00-test_seq-09-epoch-20.pth --optim_dir /home/mscv/npc/SemiSupervisedDPC/results/offline-mode/May03_ssim/optimizer-2020-5-3-0-45-val_seq-00-test_seq-09-epoch-20.pth
