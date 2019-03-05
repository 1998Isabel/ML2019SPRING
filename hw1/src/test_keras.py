import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation

def load_data(path):
    data = pd.read_csv(path, encoding='big5', header=None).values
    data = data[:, 2:]
    data = np.reshape(data, (-1,18,9)) # (240,18,9)
    features = [2, 3, 5, 6, 7, 8, 9, 10, 14, 15, 12, 16, 17]
    data = data[:,features,:]
    data[data == 'NR'] = '0.0'
    data = data.astype(float)

    X = []
    for i in range(data.shape[0]):
        one_data = data[i,:,:].flatten()
        # one_data = np.append(one_data, 1.0)
        X.append(one_data)
    
    X_test = np.array(X)
    print(X_test.shape)
    
    return X_test

def scaling(scale_path, X):
    scale = np.load(scale_path)
    Max = scale[0]
    Min = scale[1]
    X_data = (X - Min)/(Max - Min + 1e-20) + 1e-10
    return X_data

def predict(model_path, X):
    model = load_model(model_path)
    Y = model.predict(X)
    return Y

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.npy')
    # parser.add_argument('--scale_path', type=str, default='model.npy')
    parser.add_argument('--data_path', type=str, default='test.csv', help='path to data')
    parser.add_argument('--output_path', type=str, default='output.csv', help='path to output')
    opts = parser.parse_args()
    X = load_data(opts.data_path)
    # X = scaling(opts.scale_path, X)
    Y = predict(opts.model_path, X)
    Y = Y.astype(float)
    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','value'])
        for i in range(len(Y)):
            id_name = 'id_' + str(i)
            writer.writerow([id_name, float(Y[i])])
    print(Y)