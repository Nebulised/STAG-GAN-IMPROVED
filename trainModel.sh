#!/bin/bash

IMAGE_SIZE=64
NUMBER_CLASSES=6
SAMPLE_EVERY=2500
PRINT_EVERY=100


python3 main.py --img_size "$IMAGE_SIZE"\
 --num_domains "$NUMBER_CLASSES"\
 --grayscale\
 --w_hpf 0\
 --ada\
 --train_img_dir "$1"\
 --val_img_dir "$2"\
 --mode "train"\
 --sample_every "$SAMPLE_EVERY"\
 --print_every "$PRINT_EVERY"\
 --batch_size 6\
 --save_every 2500\
 --ds_iter 15000\
 --total_iters 15000\
 --lambda_cyc 5\
 --lambda_ds 2\
 --val_batch_size 4\
 --attentionGuided
