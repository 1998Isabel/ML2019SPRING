#!/bin/sh
# $1 = testing data $2 = prediction file

# Download model to model/
# wget -O model/word2vec.model https://github.com/1998Isabel/Machine_Learning/releases/download/2/word2vec.model
# wget -O model/word2vec_iter16.model https://github.com/1998Isabel/Machine_Learning/releases/download/3/word2vec_iter16.model
wget -O model/word2vec.model https://www.dropbox.com/s/8554kdjmf3ps5lr/word2vec.model?dl=1
wget -O model/word2vec_iter16.model https://www.dropbox.com/s/cq6zvq53odc63v6/word2vec_iter16.model?dl=1
wget -O model/model.h5 https://www.dropbox.com/s/5qnltykez0j1hrj/model.h5?dl=1
wget -O model/rnn_g8.h5 https://www.dropbox.com/s/3f1w9lavxav9kl8/rnn_g8.h5?dl=1
wget -O model/rnn_g6.h5 https://www.dropbox.com/s/prieh8h00d41m0s/rnn_g6.h5?dl=1

# test <test_x file> <dict.txt.big file> <output file>
python test.py --X_path $1 --output_path $3 --dict $2 --w2v_model model/word2vec_iter16.model --w2v_model_old model/word2vec.model
