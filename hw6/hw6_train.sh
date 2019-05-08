#!/bin/sh
# $1 = <train_x file>  $2 = <train_y file>  $3 = <test_x.csv file>  $4 = <dict.txt.big file>

wget -O model/word2vec_iter16.model https://github.com/1998Isabel/Machine_Learning/releases/download/3/word2vec_iter16.model

# train RNN
!python Drive/ML/hw6/train_rnn_4.py --train $1 --test $3 --Y $2 --txt model/sentences.txt --dict $4 --wordmodel model/word2vec_iter16.model --savemodel model/rnn.h5 --epoch 60
