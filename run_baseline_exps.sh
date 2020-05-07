#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

###Baseline (SS-DPC)
python3 run_mono_dpc.py --date May03_baseline --train_seq 'all' --val_seq '00' --test_seq '09'
