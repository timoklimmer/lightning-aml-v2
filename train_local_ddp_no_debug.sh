#!/usr/bin/env bash

# note: this trains a model locally in non-debug mode. if you need to debug the training script, run the respective 
# launch configuration in VS.Code

python train.py \
    --model ImageClassifier \
    --data-module MNIST \
    --training-strategy ddp_find_unused_parameters_false \
    --max-epochs 10