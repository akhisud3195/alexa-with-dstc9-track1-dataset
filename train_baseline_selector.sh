  
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

# Knowledge selection
# use --negative_sample_method to modify the setting in params.json for this training run
# the updated parameters will be saved to {checkpoint}/params.json
# note that the default negative_sample_method for testing is "oracle"
# which filters the candidates based on the ground truth entity, so the number of candidates
# is way less than the total number of snippets
python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --negative_sample_method "all" \
    --params_file baseline/configs/selection/params.json \
    --dataroot data \
    --exp_name ks-all-${version}
