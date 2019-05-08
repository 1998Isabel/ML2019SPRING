import jieba
import argparse
import csv
import numpy as np
import pandas as pd
import emoji
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from statistics import mode

def read_data(X_path, word_model, bigdict):
    print('Reading X_test')
    data = pd.read_csv(X_path)
    X_data = data['comment'].values
    print(X_data.shape) # (12000,)
    X_words = []
    jieba.set_dictionary(bigdict)
    for i in range(len(X_data)):
        line = emoji.demojize(X_data[i])
        seg_list = list(jieba.cut(line, cut_all=False))
        X_words.append(seg_list)

    w2v_model = word2vec.Word2Vec.load(word_model)
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
    word2idx = {}

    vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 1] = vec
        word2idx[word] = i + 1

    X_vecs = []
    for doc in X_words:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        X_vecs.append(new_doc)
    X = np.array(X_vecs)
    return X

def test(X, net_model):
    print(net_model)
    PADDING_LENGTH = 50
    X_test = pad_sequences(X, maxlen=PADDING_LENGTH)
    model = load_model(net_model)
    Y_preds = model.predict(X_test)
    print("Shape:", Y_preds.shape)
    print("Sample:", Y_preds[0])
    return Y_preds

def ensemble(X, X_old):
    old_modellist = ['model/model.h5']
    modellist = ['model/rnn_g8.h5',
                'model/rnn_g6.h5'
                ]
    prediction = []
    preds = []
    for o in range(len(old_modellist)):
        nowpred = test(X_old, old_modellist[o])
        newpred = []
        for y in range(len(nowpred)):
            if nowpred[y][0] > 0.5:
                newpred.append(0)
            else:
                newpred.append(1)
        preds.append(newpred)
    for j in range(len(modellist)):
        nowpred = test(X, modellist[j])
        newpred = []
        for y in range(len(nowpred)):
            if nowpred[y][0] > 0.5:
                newpred.append(0)
            else:
                newpred.append(1)
        preds.append(newpred)
    for i in range(0,len(X)):
        # try:
        ans = mode([preds[0][i], preds[1][i], preds[2][i]])
        # print(ans)
        prediction.append(ans)
    return prediction

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_path', type=str, default='test.csv', help='path to X')
    parser.add_argument('--output_path', type=str, help='path to X')
    parser.add_argument('--dict', type=str)
    parser.add_argument('--w2v_model', type=str)
    parser.add_argument('--w2v_model_old', type=str)
    # parser.add_argument('--net_model', type=str)
    opts = parser.parse_args()
    X = read_data(opts.X_path, opts.w2v_model, opts.dict)
    X_old = read_data(opts.X_path, opts.w2v_model_old, opts.dict)
    # Y = test(X, opts.net_model)
    Y = ensemble(X, X_old)
    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(len(Y)):
            writer.writerow([str(i), str(Y[i])])
            # if Y[i][0] > 0.5:
            #     writer.writerow([str(i), str(0)])
            # else:
            #     writer.writerow([str(i), str(1)])