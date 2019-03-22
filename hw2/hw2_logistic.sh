#!/bin/sh

# train
# python src/train_logistic.py --data_path $1 --X_path $3 --Y_path $4
# test
python src/test_logistic.py --model_path logistic.npy --data_path $2  --X_path $5 --output_path $6