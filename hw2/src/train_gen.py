import os
import sys
import argparse
import csv
import math as m
import numpy as np
import pandas as pd

def load_data(X_path, Y_path):
    # data = pd.read_csv(path, encoding='big5').values # (4320,27)

    X = pd.read_csv(X_path).values  #(32561, 106)
    Y = pd.read_csv(Y_path).values  #(32561, 1)
    Y = Y.reshape((Y.shape[0],))
    # size = X.shape[0]
    # Ones = np.ones((size,1))
    # X = np.append(X, Ones, axis=1)
    X = X.astype(float)
    print(X.shape)
    return X, Y

def sigmoid(Z):
    return 1 / (1.0 + np.exp(-1 * Z))

def train(X,Y):
    size, p = X.shape
    index_1 = [] # label == 1
    index_2 = [] # label == 0
    for i in range(size):
        if Y[i] == 1:
            index_1.append(i)
        else:
            index_2.append(i)
    X_1 = X[index_1]
    X_2 = X[index_2]
    N_1 = X_1.shape[0]
    N_2 = X_2.shape[0]
    # print(X_1.shape)
    # print(X_2.shape)

    Mu_1 = np.mean(X_1, axis=0)
    Mu_2 = np.mean(X_2, axis=0)
    Sigma_1 = np.cov(X_1.transpose())
    Sigma_2 = np.cov(X_2.transpose())
    Sigma = (N_1 * Sigma_1 + N_2 * Sigma_2)/size
    print(Mu_1.shape)
    print(Mu_2.shape)
    print(Sigma_1.shape)
    print(Sigma_2.shape)
    print(N_1,N_2)
    return Mu_1, Mu_2, Sigma, N_1, N_2

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
    # X, Mean, Std = scaling(X)
    Mu_1, Mu_2, Sigma, N_1, N_2 = train(X, Y)
    Mus = np.vstack((Mu_1, Mu_2))
    N = np.ones(2)
    N[0] = N_1
    N[1] = N_2
    np.save("Mu.npy",Mus)
    np.save("Sigma.npy",Sigma)
    np.save("N.npy",N)
    # print(model.shape)