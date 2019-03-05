import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation

def load_data(path):
    data = pd.read_csv(path, encoding='big5').values # (4320,27)
    data = data[:, 3:] # (4320,24)
    data = np.reshape(data, (12,-1,18,24)) #(12_months,20_days,18_features,24_hours)
    data = data.swapaxes(1,2) #(12,18,20,24)
    data = np.reshape(data, (12,18,-1)) #(12,18,480)
    features = [2, 3, 5, 6, 7, 8, 9, 10, 14, 15, 12, 16, 17]
    data = data[:,features,:]
    data[data == 'NR'] = '0.0'
    data = data.astype(float)

    X = []
    Y = []
    for m in range(12):
        for h in range(471):
            one_data = data[m,:,h:h+9].flatten()
            # one_data = np.append(one_data, 1.0)
            X.append(one_data)
            Y.append(data[m,6,h+9])
    
    X_data = np.array(X)
    Y_data = np.array(Y)
    return X_data, Y_data

def scaling(X):
    Max = np.max(X, axis=0)
    Min = np.min(X, axis=0)
    X_data = (X - Min)/(Max - Min + 1e-20) + 1e-20
    # print(X_data)
    return X_data, Max, Min

def train(X,Y):
    print(X)
    print(Y)
    feat_dim = X.shape[1]
    model = Sequential()
    model.add(Dense(units=1, input_dim=feat_dim, use_bias=True))
    model.compile(loss='mean_squared_error', optimizer="Adam")
    model.fit(X, Y, batch_size=256, epochs=1024, verbose=1)
    model.save('model_keras3_Adam_scale')
    return

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to data')
    opts = parser.parse_args()
    #print(opts.data_path)
    X, Y = load_data(opts.data_path)
    # X, Max, Min = scaling(X)
    train(X,Y)
    # scale = np.vstack((Max, Min))
    # np.save("scaling.npy",scale)
    #print(w.shape)