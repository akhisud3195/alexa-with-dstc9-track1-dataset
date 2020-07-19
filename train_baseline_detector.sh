#!/bin/bash

# This script demonstrates how to train baseline models with this repo
# We train models for three subtasks separately
# 1. knowledge-seeking turn detection
# 2. knowledge selection
# 3. response generation
# And we show how to generate responses for test dataset without labels.json at the end

# set path to dataset here
version="baseline"
dataroot="data"
num_gpus=1

# Knowledge-seeking turn detection
# distributed training, single-process multi-gpu training also supported
# use --params_file to specify the file containing training parameters
# use --exp_name to specify the name of this run, the checkpoints and logs will be stored in runs/{exp_name}
# use --eval_desc to specify the description of evaluation, which will be written in eval_results.txt

python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
        --params_file baseline/configs/detection/params.json \
        --dataroot data \
        --exp_name ktd-${version}
