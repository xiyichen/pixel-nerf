#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/scratch/xiychen/miniconda3/lib/
module load cuda/11.1.1
python train/train.py -n facescape_exp -c conf/exp/facescape.conf -D /cluster/scratch/xiychen/data/facescape_color_calibrated --gpu_id=0 -B 4