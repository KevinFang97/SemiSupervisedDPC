#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)_smo

###Baseline (SS-DPC) + ssim + smo
python3 run_mono_dpc.py --date May05_smo_su --val_seq '00' --test_seq '09' --num_epochs 9 --lr 3e-5 --wd 3e-6 --lr_decay_epoch 3 --supervised --model_dir /home/mscv/npc/SemiSupervisedDPC/results/offline-mode/May03_smo/2020-5-3-15-53-val_seq-00-test_seq-09-epoch-16.pth --optim_dir /home/mscv/npc/SemiSupervisedDPC/results/offline-mode/May03_smo/optimizer-2020-5-3-15-53-val_seq-00-test_seq-09-epoch-16.pth
