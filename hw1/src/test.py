import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path,header=None)
    data = data.drop(data.columns[[0, 1]], axis=1)
    data = data.values
    train_data = np.array(data)
    row, col = train_data.shape
    #print(train_data.shape)
    size = int(row/18)

    X = []
    for _ in range(size):
        X.append([])
    
    for s in range(size):
        for f in range(18):
            for h in range(col):
                n = s * size + f
                if n < row:
                    if train_data[n][h] == 'NR':
                        a = 0.0
                    else:
                        a = float(train_data[n][h])
                X[s].append(a)
        X[s].append(1.0)
    X_np = np.array(X)
    return X_np

def predict(model_path, X):
    model = np.load(model_path)
    # print(model)
    Y = np.inner(X, model)
    return Y

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.npy')
    parser.add_argument('--data_path', type=str, default='test.csv', help='path to data')
    parser.add_argument('--output_path', type=str, default='output.csv', help='path to output')
    opts = parser.parse_args()
    X = load_data(opts.data_path)
    Y = predict(opts.model_path, X)
    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','value'])
        for i in range(len(Y)):
            id_name = 'id_' + str(i)
            writer.writerow([id_name, Y[i]])
    #print(Y)