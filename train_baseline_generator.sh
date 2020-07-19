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
num_gpus=4

# Response generation
python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --params_file baseline/configs/generation/params.json \
    --dataroot data \
    --exp_name rg-hml128-kml128-${version}
