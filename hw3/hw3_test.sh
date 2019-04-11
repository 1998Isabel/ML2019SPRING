#!/bin/sh
# $1 = testing data $2 = prediction file

# Download model to model/
wget -O model/cnn.h5 https://github.com/1998Isabel/Machine_Learning/releases/download/0.0.0/cnn.h5

# test
python test.py $1 $2 model/cnn.h5
