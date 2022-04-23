#!/bin/bash

IMAGE_SIZE=64
NUMBER_CLASSES=9
SAMPLE_EVERY=500


python3 main.py --img_size "$IMAGE_SIZE"\
 --num_domains "$NUMBER_CLASSES"\
 --w_hpf 0\
 --grayscale\
 --randcrop_prob 0.0\
 --train_img_dir "$1"\
 --val_img_dir "$2"\
 --mode "train"\
 --sample_every "$SAMPLE_EVERY"\

