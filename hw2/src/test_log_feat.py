import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd

def nation_map(path, X_nation):
    data = pd.read_csv(path, header=None)
    data[41][0] = ' ?'
    nation_dic = {data[i][0]: float(data[i][1]) for i in range(data.shape[1])}
    for i in range(len(X_nation)):
        X_nation[i] = nation_dic.get(X_nation[i])
    # print(X_nation)
    return X_nation

def load_rawdata(data_path, X_path, nation_path):
    data = pd.read_csv(data_path).values
    X = pd.read_csv(X_path).values
    size = X.shape[0]
    X_data = np.copy(X)
    for i in range(41):
        X_data = np.delete(X_data, 105-i, 1) # delete nation
    # for i in range(16):
    #     X = np.delete(X, 64-i, 1) # delete educaton
    unknown = [14,52]
    X_data = np.delete(X_data, unknown, 1)
    X_data = np.delete(X_data, 1, 1) # delete fnlwgt
    print(X_data.shape)
    index_strong = [65,66,72,73,74,87,102] # add strong nation back
    index_mediam = [75,78,80,81,84,89,91,94,97,99]
    X_strongnation = X[:,index_strong]
    Strongnation = np.sum(X_strongnation, axis=1).reshape((size,1))
    X_data = np.concatenate((X_data, Strongnation), 1)
    X_mediamnation = X[:,index_mediam]
    Mediamnation = np.sum(X_mediamnation, axis=1).reshape((size,1))
    X_data = np.concatenate((X_data, Mediamnation), 1)
    print(X_data.shape)

    nations = data[:,13]
    nations = nation_map(nation_path, nations)
    nations = np.expand_dims(nations, 1)
    X_data = np.concatenate((X_data, nations), 1)
    print(X_data.shape) # (32561, 65)

    data = data[:,4] # education_num
    data = np.expand_dims(data, 1)
    print(data.shape)
    X_data = np.concatenate((X_data, data), 1)
    size = X_data.shape[0]
    Ones = np.ones((size,1))
    X_data = np.concatenate((X_data, Ones), 1)
    X_data = X_data.astype(float)
    print(X_data.shape)
    # with open('X_test_new.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(X)
    return X_data

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
    parser.add_argument('--X_path', type=str, default='X_test', help='path to X_test')
    parser.add_argument('--output_path', type=str, default='output.csv', help='path to output')
    parser.add_argument('--nation_path', type=str, default='Nation.csv', help='path to Nation')
    opts = parser.parse_args()
    # X = load_data(opts.data_path)
    X = load_rawdata(opts.data_path, opts.X_path, opts.nation_path)
    X, w = scaling(opts.model_path, X)
    Y = predict(w, X)
    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(len(Y)):
            # id_name = 'id_' + str(i)
            writer.writerow([i+1, Y[i]])
    print(Y)