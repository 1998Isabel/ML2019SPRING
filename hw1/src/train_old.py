import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_data(path):
    data = pd.read_csv(path)
    data = data.drop(data.columns[[0, 1, 2]], axis=1)
    data = data.values
    row, col = data.shape
    
    train_data = [[] for i in range(18)]
    X = [[] for i in range(163)]
    Y = []
    
    for i in range(row):
        for j in range(col):
            train_data[i%18].append(data[i][j])
    train_data = np.array(train_data)

    # for debug: write train_data
    # with open('train_data', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(train_data)
    # print(train_data.shape)

    for i in range(12):
        for j in range(480):
            if j >= 9:
                d = i*480 + j
                for k in range(18):
                    for l in range(9):
                        f = k * 9 + l
                        if train_data[k][d-9+l] == 'NR':
                            a = 0.0
                        else:
                            a = float(train_data[k][d-9+l])
                        X[f].append(a)
                X[162].append(1.0)
                Y.append(float(train_data[9][d]))
    X_data = np.array(X)
    X_data = np.transpose(X_data)
    Y_data = np.array(Y)
    
    # for debug: write train_data
    # with open('X_data.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(X_data)
    # print(X_data.shape)
    # print(Y_data.shape)

    # feature, size, hrs = X_data.shape
    # X_range = []
    # for i in range(size):
    #     X_range.append([])
    
    # for s in range(size):
    #     for f in range(feature):
    #         for h in range(hrs):
    #             if X_data[f][s][h] == 'NR':
    #                 a = 0
    #             else:
    #                 a = float(X_data[f][s][h])
    #             X_range[s].append(a)
    #     X_range[s].append(1.0)
    # X_range_np = np.array(X_range)
    # # print(X_range_np.shape)
    # # print(Y_data.shape)

    return X_data, Y_data

def scaling(X):
    X_scaled = preprocessing.scale(X)
    return X_scaled, X_scaled.mean(axis=0), X_scaled.std(axis=0)

def train(X,Y):
    size, p = X.shape
    # print(p)
    epochs = 100000
    lr = 0.5
    w = np.zeros(p)
    prev_grad = np.zeros(p)
    x_t = X.transpose()
    
    for i in range(epochs):
        y_temp = np.dot(X,w)
        #print(w.shape)
        Loss = y_temp - Y
        grad = 2 * np.dot(x_t,Loss)
        # print(grad)
        prev_grad += grad**2
        adagrad = np.sqrt(prev_grad)
        w -= lr * grad / adagrad
        # print(i)
    # print(w)
    return w


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to data')
    opts = parser.parse_args()
    #print(opts.data_path)
    X, Y = load_data(opts.data_path)
    # X_scaled, Mean, Std = scaling(X)
    model = train(X,Y)
    # print(w)
    # model = np.vstack((X_scaled, Mean, Std))
    np.save("model_6.npy",model)
    print(model.shape)

    # for debug: write train_data
    # with open('X_scaled.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(model)