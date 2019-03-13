import os
import sys
import argparse
import csv
import math as m
import numpy as np
import pandas as pd

def load_data(X_path, Y_path):
    # data = pd.read_csv(path, encoding='big5').values # (4320,27)
    # data = data[:, 3:] # (4320,24)
    # data = np.reshape(data, (12,-1,18,24)) #(12_months,20_days,18_features,24_hours)
    # data = data.swapaxes(1,2) #(12,18,20,24)
    # data = np.reshape(data, (12,18,-1)) #(12,18,480)
    # data = np.delete(data, 10, 1) # delete RAIN_FALL (12,17,480)
    # data = data.astype(float)

    X = pd.read_csv(X_path).values  #(32561, 106)
    Y = pd.read_csv(Y_path).values  #(32561, 1)
    Y = Y.reshape((Y.shape[0],))
    size = X.shape[0]
    Ones = np.ones((size,1))
    X = np.append(X, Ones, axis=1)
    X = X.astype(float)
    print(X.shape)
    return X, Y

def sigmoid(Z):
    return 1 / (1.0 + np.exp(-1 * Z))

def train(X,Y):
    print(X.shape)
    size, p = X.shape
    epochs = 3000
    lr = 1
    w = np.zeros((p,))
    # b = np.zeros((1,))
    prev_w_grad = np.zeros((p,))
    # prev_b_grad = np.zeros((1,))
    x_t = np.transpose(X)
    
    for T in range(epochs):
        Z = np.dot(X, w)#  + b
        f_x = sigmoid(Z)
        Cross = -(np.dot(Y, np.log(f_x)) + np.dot((1 - Y), np.log(1 - f_x)))
        
        Loss = f_x - Y
        w_grad = np.sum(X * Loss.reshape((size,1)), axis=0)
        # b_grad = np.sum(Loss)
        prev_w_grad += w_grad**2
        # prev_b_grad += b_grad**2
        w_adagrad = np.sqrt(prev_w_grad)
        # b_adagrad = np.sqrt(prev_b_grad)
        w -= lr * w_grad / w_adagrad
        # b -= lr * b_grad / b_adagrad
        if T%100 == 0:
            print("Iteration ", T, ": ", Cross)
    # w = np.concatenate(w, b[0])
    return w

def scaling(X):
    Mean = np.mean(X, axis=0)
    Std = np.std(X, axis=0)
    # print(Mean.shape)
    index = [0, 1, 3, 4, 5]
    Means = np.zeros(X.shape[1])
    Stds = np.ones(X.shape[1])
    Means[index] = Mean[index]
    Stds[index] = Std[index]
    X_data = (X - Means)/Stds
    # print(X_data)
    return X_data, Means, Stds

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_path', type=str, default='X_train', help='path to X')
    parser.add_argument('--Y_path', type=str, default='Y_train', help='path to Y')
    opts = parser.parse_args()
    
    X, Y = load_data(opts.X_path, opts.Y_path)
    X, Mean, Std = scaling(X)
    w = train(X, Y)
    model = np.vstack((w, Mean, Std))
    np.save("logistic.npy",model)
    print(model.shape)