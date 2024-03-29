import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path, encoding='big5', header=None).values
    data = data[:, 2:]
    data = np.reshape(data, (-1,18,9)) # (240,18,9)
    
    # extract features
    # features = [2, 3, 5, 6, 7, 8, 9, 10, 14, 15, 12, 16, 17]
    # features = [3, 5, 6, 7, 8, 9, 12, 16, 17]
    # data = data[:,features,:]
    data[data == 'NR'] = '0.0'
    data = data.astype(float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i][j][k] < 0 and k > 0 and k < data.shape[2]-1:
                    prevone = k - 1
                    nextone = k + 1
                    days = 2
                    while data[i][j][nextone] < 0:
                        nextone = nextone + 1
                        days = days + 1
                    if nextone >= data.shape[2]:
                        data[i][j][k] = 0.0
                    else:
                        data[i][j][k] = (data[i][j][prevone]*(days-1) + data[i][j][nextone])/days                   
    data[data < 0.0] = 0.0


    X = []
    for i in range(data.shape[0]):
        one_data = data[i,:,:].flatten()
        one_data = np.append(one_data, 1.0)
        X.append(one_data)
    
    X_test = np.array(X)
    # print(X_test.shape)
    
    return X_test

def predict(model_path, X):
    w = np.load(model_path)
    Y = np.dot(X,w)
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