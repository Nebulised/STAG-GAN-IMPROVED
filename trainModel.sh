#!/bin/bash

imageSize=64
numberClasses=3
SAMPLE_EVERY=500


python3 main.py --img_size "$IMAGE_SIZE"\
 --num_domains "$numberClasses"\
 --w_hpf 0\
 --randcrop_prob 0.0\
 --train_img_dir "$1"\
 --val_img_dir "$2"\
 --mode "train"\
 --sample_every "$SAMPLE_EVERY"
