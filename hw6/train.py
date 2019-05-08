import jieba
import argparse
import numpy as np
import pandas as pd
import emoji
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, LSTM, Dense, Dropout, Activation, Conv1D, MaxPooling1D
from keras.callbacks import CSVLogger, ModelCheckpoint


def read_data(X_train, X_test, Y_path, sentence_txt, bigdict, word2vec_model):
    TRAIN_NUM = 119018
    try:
        print('Loading Sentences')
        sentences = word2vec.LineSentence(sentence_txt)
    except:
        print('Reading data to sentences')
        data = pd.read_csv(X_train)
        X_data = data['comment'].values
        testdata = pd.read_csv(X_test)
        X_testdata = testdata['comment'].values
        print(X_data.shape) # (12000,)
        print(X_testdata.shape)
        X_words = []
        jieba.set_dictionary(bigdict)
        for i in range(len(X_data)):
            line = emoji.demojize(X_data[i])
            seg_list = list(jieba.cut(line, cut_all=False))
            X_words.append(seg_list)
        for j in range(len(X_testdata)):
            line = emoji.demojize(X_testdata[j])
            seg_list = list(jieba.cut(line, cut_all=False))
            X_words.append(seg_list)
        
        out = open(sentence_txt, "w")
        for sen in X_words:
            for word in sen:
                out.write(word)
                out.write(' ')
            out.write('\n')
        out.close()
        sentences = word2vec.LineSentence(sentence_txt)

    # word2vec
    try:
        print('Loading word2vec model')
        w2v_model = word2vec.Word2Vec.load(word2vec_model)
    except:
        print('Training word2vec model')
        w2v_model = word2vec.Word2Vec(sentences, iter=32, size=128, min_count=3, workers=4, sg=1)
        w2v_model.save(word2vec_model)

    embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
    word2idx = {}

    vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
    for v, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[v + 1] = vec
        word2idx[word] = v + 1
    
    global embedding_layer
    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)
    X_vecs = []
    readfile = open(sentence_txt, "r")
    for line in readfile:
        new_doc = []
        for word in line.split():
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        X_vecs.append(new_doc)
        if len(X_vecs) >= TRAIN_NUM:
            break
    X = np.array(X_vecs)
    print(X.shape)

    label = pd.read_csv(Y_path)
    Y_data = label['label'].values
    Y = np.array(Y_data)
    Y = Y[0:TRAIN_NUM]
    
    return X, Y

def train(X, Y, savemodel, epoch_num):
    BATCH_SIZE = 3000
    EPOCHS = int(epoch_num)
    PADDING_LENGTH = 50
    X = pad_sequences(X, maxlen=PADDING_LENGTH)
    print("Shape:", X.shape) # (120000, 50)
    Y = to_categorical(Y)
    print("Shape:", Y.shape) # (120000, 2)

    model = new_model()
    model.summary()
    # Callbacks
    callbacks = []
    modelcheckpoint = ModelCheckpoint('Drive/ML/hw6/model_2/weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
    callbacks.append(modelcheckpoint)
    model.fit(x=X, y=Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, callbacks=callbacks,)
    model.save(savemodel)

def new_model():
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, go_backwards=True))    
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, go_backwards=True))    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/train_x.csv', help='path to X')
    parser.add_argument('--test', type=str, default='data/test_x.csv', help='path to X')
    parser.add_argument('--Y', type=str, default='data/train_y.csv', help='path to Y')
    parser.add_argument('--txt', default='sentences.txt', type=str)
    parser.add_argument('--dict', default='dict.txt.big', type=str)
    parser.add_argument('--wordmodel', default='word2vec_all.model', type=str)
    parser.add_argument('--savemodel', type=str)
    parser.add_argument('--epoch', type=str)
    opts = parser.parse_args()
    X, Y = read_data(opts.train, opts.test, opts.Y, opts.txt, opts.dict, opts.wordmodel)
    # X, Y = load_data()
    train(X, Y, opts.savemodel, opts.epoch)