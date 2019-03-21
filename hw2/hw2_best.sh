#!/bin/sh

# train
# python src/train_log_feat.py --data_path $1 --X_path $3 --Y_path $4
# test
python src/test_log_feat.py --model_path logistic_raw_14.npy --data_path $2  --X_path $5 --output_path $6