import os
import sys
import argparse
import csv
import math as m
import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path, encoding='big5', header=None).values
    data = data[1:,:]
    data = data.astype(float)
    size = data.shape[0]
    return data

def sigmoid(Z):
    return 1 / (1.0 + np.exp(-1 * Z))

def scaling(model_path, X):
    model_all = np.load(model_path)
    with open('Model.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(model_all)
    w = model_all[0]
    Mean = model_all[1]
    Std = model_all[2]
    
    X = (X - Mean)/Std
    return X, w

def predict(Mu_path, Sigma_path, N_path, X):
    Mus = np.load(Mu_path)
    Mu_1 = Mus[0]
    Mu_2 = Mus[1]
    Sigma = np.load(Sigma_path)
    N = np.load(N_path)
    N_1 = N[0]
    N_2 = N[1]

    w = np.dot((Mu_1 - Mu_2).T, np.linalg.inv(Sigma)).T
    b = -0.5 * np.dot(np.dot(Mu_1.T, np.linalg.inv(Sigma)), Mu_1)
    b += 0.5 * np.dot(np.dot(Mu_2.T, np.linalg.inv(Sigma)), Mu_2)
    print(N_1)
    print(N_2)
    b += np.log(float(N_1)/N_2)

    Z = np.dot(X,w) + b
    Y = sigmoid(Z)
    Y[Y > 0.5] = 1
    Y[Y <= 0.5] = 0
    Y = Y.astype(int)
    return Y

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu_path', type=str, default='Mu.npy')
    parser.add_argument('--sigma_path', type=str, default='Sigma.npy')
    parser.add_argument('--n_path', type=str, default='N.npy')
    parser.add_argument('--data_path', type=str, default='test.csv', help='path to data')
    parser.add_argument('--output_path', type=str, default='output.csv', help='path to output')
    opts = parser.parse_args()
    X = load_data(opts.data_path)
    # X, w = scaling(opts.model_path, X)
    Y = predict(opts.mu_path, opts.sigma_path, opts.n_path, X)
    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(len(Y)):
            # id_name = 'id_' + str(i)
            writer.writerow([i+1, Y[i]])
    # print(Y)