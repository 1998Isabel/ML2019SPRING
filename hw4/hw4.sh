#!/bin/sh
# $1 = training data $2 = output path

# download model
wget -O model/cnn.h5 https://github.com/1998Isabel/Machine_Learning/releases/download/0.0.0/cnn.h5

# saliency map
python saliency_map.py --data $1 --model model/cnn.h5 --output $2

# filter activate
python filters_activate.py --model model/cnn.h5 --output $2
# output
python filters.py --model model/cnn.h5 --data $1 --output $2

# lime
python lime_img.py --data $1 --model model/cnn.h5 --output $2 