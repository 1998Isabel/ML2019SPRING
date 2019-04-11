#!/bin/sh
# $1 = training data

python train.py --data_path $1 --model_path model/cnn.h5