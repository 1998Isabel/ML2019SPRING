#!/bin/sh
# bash  hw8_train.sh <training data>
# $1 = <training data>

mkdir model/weights

# train
python src/train_mobile.py --data_path $1 --weights weights
