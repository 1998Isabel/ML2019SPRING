#!/bin/sh
# bash cluster.sh <images path> <test_case.csv path> <prediction file path>
# $1 = <images path> $2 = <test_case.csv path> $3 = <prediction file path>

# Download autoencoder to model/
wget -O model/encoder.h5 https://www.dropbox.com/s/vfmda036mz5tze1/automodel_dnn_5.h5?dl=1

# test 
python src/cluster.py --test $2 --input $1 --encoder model/encoder.h5 --output $3
