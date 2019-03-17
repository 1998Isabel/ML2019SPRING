#!/bin/sh

# train
python src/train_gen.py --X_path $3 --Y_path $4
# test
python src/test_gen.py --data_path $5 --output_path $6