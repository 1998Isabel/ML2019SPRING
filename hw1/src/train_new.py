import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path, encoding='big5').values # (4320,27)
    data = data[:, 3:] # (4320,24)
    data = np.reshape(data, (12,-1,18,24)) #(12_months,20_days,18_features,24_hours)
    data = data.swapaxes(1,2) #(12,18,20,24)
    data = np.reshape(data, (12,18,-1)) #(12,18,480)
    
    # extract features
    features = [2, 3, 5, 6, 7, 8, 9, 10, 14, 15, 12, 16, 17]
    # features = [3, 5, 6, 7, 8, 9, 12, 16, 17]
    data = data[:,features,:]
    print(data.shape)
    data[data == 'NR'] = '0.0'
    data = data.astype(float)
    data[data < 0.0] = 0.0

    X = []
    Y = []
    for m in range(12):
        for h in range(471):
            one_data = data[m,:,h:h+9].flatten()
            one_data = np.append(one_data, 1.0)
            X.append(one_data)
            Y.append(data[m,6,h+9]) # needs to change if extract features
    
    X_data = np.array(X)
    Y_data = np.array(Y)
    return X_data, Y_data

def train(X,Y):
    print(X.shape)
    size, p = X.shape
    # print(p)
    epochs = 100000
    lr = 1
    w = np.zeros(p)
    prev_grad = np.zeros(p)
    x_t = X.transpose()
    
    for _ in range(epochs):
        y_temp = np.dot(X,w)
        #print(w.shape)
        Loss = y_temp - Y
        grad = 2 * np.dot(x_t,Loss)
        prev_grad += grad**2
        adagrad = np.sqrt(prev_grad)
        w -= lr * grad / adagrad
    print(w.shape)
    return w

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to data')
    opts = parser.parse_args()
    #print(opts.data_path)
    X, Y = load_data(opts.data_path)
    w = train(X,Y)
    # print(w)
    np.save("model_zero_feat_long.npy",w)
    #print(w.shape)