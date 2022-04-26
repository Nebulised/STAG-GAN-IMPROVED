#!/bin/bash

IMAGE_SIZE=128
NUMBER_CLASSES=6
SAMPLE_EVERY=2500
PRINT_EVERY=2500


python3 main.py --img_size "$IMAGE_SIZE"\
 --num_domains "$NUMBER_CLASSES"\
 --w_hpf 0\
 --grayscale\
 --randcrop_prob 0.5\
 --train_img_dir "$1"\
 --val_img_dir "$2"\
 --mode "train"\
 --sample_every "$SAMPLE_EVERY"\
 --print_every "$PRINT_EVERY"\
 --batch_size 3\
 --save_every 5000\
 --ds_iter 30000\
 --total_iters 30000

