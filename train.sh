#!/bin/bash
# rm -r logs
# rm -r checkpoints
# rm -r visuals
python train/train.py -n facescape_exp -c conf/exp/facescape.conf -D /root/data/facescape_color_calibrated --gpu_id=0 -B 4
# 1790163