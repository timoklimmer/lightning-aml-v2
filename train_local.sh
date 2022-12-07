#!/usr/bin/env bash
python train.py \
    --model ImageClassifier \
    --data-module MNIST \
    --training-strategy ddp_find_unused_parameters_false \
    --max-epochs 100