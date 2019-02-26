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
    test_data = np.array(data)
    row, col = test_data.shape
    #print(test_data.shape)
    size = int(row/18) # size = 240

    X = np.ones(shape=(size,163))

    for s in range(size):
        for k in range(18):
            for l in range(9):
                n = s * 18 + k
                f = k * 9 + l
                if test_data[n][l] == 'NR':
                    a = 0.0
                else:
                    a = float(test_data[n][l])
                    X[s][f] = a
    
    # for s in range(size):
    #     for f in range(18):
    #         for h in range(col):
    #             n = s * size + f
    #             if n < row:
    #                 if train_data[n][h] == 'NR':
    #                     a = 0.0
    #                 else:
    #                     a = float(train_data[n][h])
    #             X[s].append(a)
    #     X[s].append(1.0)

    X_test = np.array(X)
    # # for debug: write train_data
    # with open('X_test.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(X_test)
    
    # print(X_test.shape)
    return X_test

def predict(model_path, X):
    model = np.load(model_path)
    # print(model)
    # print(X.shape)
    Y = np.dot(X,model)
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
    # print(Y)