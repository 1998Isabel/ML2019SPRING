import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    data = data.drop(data.columns[[0, 1, 2]], axis=1)
    data = data.values
    row, col = data.shape
    
    train_data = []
    X = []
    Y = []
    for i in range(18):
        train_data.append([])
        X.append([])
    for i in range(row):
        for j in range(col):
            train_data[i%18].append(data[i][j])
    train_data = np.array(train_data)
    # print(train_data.shape)

    for i in range(12):
        for j in range(480):
            if j >= 9:
                d = i*480 + j
                for k in range(18):
                    X[k].append(train_data[k][d-9:d])
                Y.append(float(train_data[9][d]))
    X_data = np.array(X)
    Y_data = np.array(Y)
    # print(X_data.shape)
    # print(Y_data.shape)
    feature, size, hrs = X_data.shape
    X_range = []
    for i in range(size):
        X_range.append([])
    
    for s in range(size):
        for f in range(feature):
            for h in range(hrs):
                if X_data[f][s][h] == 'NR':
                    a = 0
                else:
                    a = float(X_data[f][s][h])
                X_range[s].append(a)
        X_range[s].append(1.0)
    X_range_np = np.array(X_range)
    # print(X_range_np.shape)
    # print(Y_data.shape)

    return X_range_np, Y_data

def train(X,Y):
    size, p = X.shape
    # print(p)
    epochs = 10000
    lr = 1
    w = np.zeros(p)
    pre_grad = np.zeros(p)
    x_t = X.transpose()
    
    for _ in range(epochs):
        y_temp = np.dot(X,w)
        #print(w.shape)
        Loss = y_temp - Y
        grad = 2 * np.dot(x_t,Loss)
        pre_grad += grad**2
        adagrad = np.sqrt(pre_grad)
        w -= lr * grad / adagrad
    #print(w)
    return w


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to data')
    opts = parser.parse_args()
    #print(opts.data_path)
    X, Y = load_data(opts.data_path)
    w = train(X,Y)
    #print(w)
    np.save("model.npy",w)
    #print(w.shape)