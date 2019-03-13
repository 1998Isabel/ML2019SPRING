import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path, encoding='big5', header=None).values
    data = data[1:,:]
    data = data.astype(float)
    size = data.shape[0]
    Ones = np.ones((size,1))
    data = np.append(data, Ones, axis=1)
    print(data.shape)
    print(data)
    
    return data

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

def predict(w, X):
    Y = np.dot(X,w)
    Y[Y > 0] = 1
    Y[Y <= 0] = 0
    Y = Y.astype(int)
    return Y

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.npy')
    parser.add_argument('--data_path', type=str, default='test.csv', help='path to data')
    parser.add_argument('--output_path', type=str, default='output.csv', help='path to output')
    opts = parser.parse_args()
    X = load_data(opts.data_path)
    X, w = scaling(opts.model_path, X)
    Y = predict(w, X)
    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(len(Y)):
            # id_name = 'id_' + str(i)
            writer.writerow([i+1, Y[i]])
    print(Y)