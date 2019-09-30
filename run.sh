#!/usr/bin/env bash

# train the model
python train.py --embed_dir data/embed --label_dir data/text --training_sets $1 --out_dir $2

# test the model on PDTB3 test set and Ted-MDB corpus
python eval.py --model_dir $2 --target pdtb3 ted
