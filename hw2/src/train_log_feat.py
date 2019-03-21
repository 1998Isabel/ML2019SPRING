import os
import sys
import argparse
import csv
import math as m
import numpy as np
import pandas as pd

def nation_map(path, X_nation):
    data = pd.read_csv(path, header=None)
    data[41][0] = ' ?'
    nation_dic = {data[i][0]: float(data[i][1]) for i in range(data.shape[1])}
    notused = []
    for i in range(len(X_nation)):
        if X_nation[i] == ' ?':
            notused.append(i)
        X_nation[i] = nation_dic.get(X_nation[i])
    # print(X_nation)
    return X_nation, notused

def age_class(data):
    size = len(data)
    ages = np.zeros((size, 9))
    for i in range(size):
        group = int(data[i]/10) -1
        if group >= 9 :
            group = 8
        if group <= 0 :
            group = 0
        ages[i][group] = 1
    # print(ages.shape)
    return ages

def load_rawdata(data_path, X_path, Y_path, nation_path):
    data = pd.read_csv(data_path).values
    X = pd.read_csv(X_path).values
    size = X.shape[0]
    Y = pd.read_csv(Y_path).values  #(32561, 1)
    Y = Y.reshape((Y.shape[0],))
    X_data = np.copy(X)
    for i in range(42):
        X_data = np.delete(X_data, 105-i, 1) # delete nation
    # for i in range(16):
    #     X_data = np.delete(X_data, 64-i, 1) # delete educaton
    unknown = [14,52]
    X_data = np.delete(X_data, unknown, 1)
    X_data = np.delete(X_data, 1, 1) # delete fnlwgt
    print(X_data.shape) # (32561, 61)

    index_strong = [65,66,72,73,74,80,87,102] # add strong nation back
    X_strongnation = X[:,index_strong]
    Strongnation = np.sum(X_strongnation, axis=1).reshape((size,1))
    X_data = np.concatenate((X_data, Strongnation), 1)
    index_mediam = [75,78,80,81,84,89,91,94,97,99] # add mediam nation back
    X_mediamnation = X[:,index_mediam]
    Mediamnation = np.sum(X_mediamnation, axis=1).reshape((size,1))
    X_data = np.concatenate((X_data, Mediamnation), 1)
    print(X_data.shape) # (32561, 63)

    nations = data[:,13]
    nations, notused = nation_map(nation_path, nations)
    nations = np.expand_dims(nations, 1)
    X_data = np.concatenate((X_data, nations), 1)
    print(X_data.shape) # (32561, 64)

    # add age class
    ages = age_class(data[:,0])
    X_data = np.concatenate((X_data, ages), 1)
    print(X_data.shape) # (32561, 73)

    data = data[:,4] # education_num
    data = np.expand_dims(data, 1)
    X_data = np.concatenate((X_data, data), 1)
    size = X_data.shape[0]
    print(X_data.shape) # (32561, 74)

    # add square, cubic
    square_sieve = [0, 2, 3, 4, 63, 73]
    cubic_sieve = [0, 2, 3, 4, 63, 73]
    Squ = X_data[:,square_sieve]
    Cu = X_data[:,cubic_sieve]
    X_data = np.concatenate((X_data, Squ**2), 1)
    X_data = np.concatenate((X_data, Cu**3), 1)
    print(X_data.shape) # (32561, 86)

    # add bias
    Ones = np.ones((size,1))
    X_data = np.concatenate((X_data, Ones), 1)
    print(X_data.shape) # (32561, 87)

    X_data = X_data.astype(float)
    Y = Y.astype(float)
    # delete not used
    X_data = np.delete(X_data, notused, axis=0)
    Y = np.delete(Y, notused, axis=0)

    with open('X_new.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(X_data)
    return X_data, Y

def load_data(X_path, Y_path):
    # data = pd.read_csv(path).values # (4320,27)
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
    X = np.concatenate((X, Ones), 1)
    X = X.astype(float)
    # print(X.shape)
    return X, Y

def sigmoid(Z):
    return np.clip(1 / (1.0 + np.exp(-Z)), 1e-6, 1-1e-6)

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
    lamdas = [0, 0.1, 0.01, 0.001, 0.0001]
    lamda = np.ones(p) * lamdas[2]
    lamda[-1] = 0.0
    
    for T in range(epochs):
        Z = np.dot(X, w)#  + b
        f_x = sigmoid(Z)
        Cross = -(np.dot(Y, np.log(f_x)) + np.dot((1 - Y), np.log(1 - f_x)))
        
        Loss = f_x - Y
        w_grad = np.sum(X * Loss.reshape((size,1)), axis=0) + np.multiply(lamda, w)
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

# def train(X,Y):
#     # print(X.shape)
#     dev_size = 0.1155
#     X, Y, X_dev, Y_dev = train_dev_split(X, Y, dev_size = dev_size)
#     num_train = len(Y)
#     num_dev = len(Y_dev)

#     size, p = X.shape
#     epochs = 2851
#     lr = 1
#     w = np.ones((p,))
#     prev_w_grad = np.zeros((p,))
#     # prev_b_grad = np.zeros((1,))
#     x_t = np.transpose(X)
#     lamdas = [0, 0.1, 0.01, 0.001, 0.0001]
#     lamda = np.ones(p) * lamdas[2]
#     lamda[-1] = 0.0
#     loss_all = []

#     loss_train = []
#     loss_validation = []
#     train_acc = []
#     dev_acc = []
#     best_train_acc = 0.0
#     best_valid_acc =0.0
#     best_t_T = 0
#     best_v_T = 0
    
#     for T in range(epochs):
#         X, Y = _shuffle(X, Y)
#         Z = np.dot(X, w)
#         # print(Z)
#         f_x = sigmoid(Z)
#         Cross = -(np.dot(Y, np.log(f_x)) + np.dot((1 - Y), np.log(1 - f_x)))
        
#         Loss = f_x - Y
#         w_grad = np.sum(X * Loss.reshape((size,1)), axis=0) + np.multiply(lamda, w)
#         # b_grad = np.sum(Loss)
#         prev_w_grad += w_grad**2
#         # prev_b_grad += b_grad**2
#         w_adagrad = np.sqrt(prev_w_grad)
#         # b_adagrad = np.sqrt(prev_b_grad)
#         w -= lr * w_grad / w_adagrad
#         # b -= lr * b_grad / b_adagrad

#         # if T%100 == 0:
#         #     print("Iteration ", T, ": ", Cross)
#         y_train_pred = get_prob(X, w)
#         Y_train_pred = np.round(y_train_pred)
#         train_acc.append(accuracy(Y_train_pred, Y))
#         loss_train.append(loss(y_train_pred, Y, lamdas[2], w)/num_train)
        
#         y_dev_pred = get_prob(X_dev, w)
#         Y_dev_pred = np.round(y_dev_pred)
#         dev_acc.append(accuracy(Y_dev_pred, Y_dev))
#         loss_validation.append(loss(y_dev_pred, Y_dev, lamdas[2], w)/num_dev)
        
#         if train_acc[T] > best_train_acc:
#             best_train_acc = train_acc[T]
#             best_t_T = T
#         if dev_acc[T] > best_valid_acc:
#             best_valid_acc = dev_acc[T]
#             best_v_T = T

#         if T%100 == 0:
#             print("Iteration ", T, ": ")
#             print("Train: ",train_acc[T], loss_train[T])
#             print("Valid: ",dev_acc[T], loss_validation[T])
#     print(best_t_T, best_train_acc, best_v_T, best_valid_acc)
#     dev_acc_n = np.array(dev_acc)
#     print(np.mean(dev_acc_n))
#     # w = np.concatenate(w, b[0])
#     return w

def scaling(X):
    Mean = np.mean(X, axis=0)
    Std = np.std(X, axis=0)
    # print(Mean.shape)
    # index = [0, 1, 3, 4, 5]
    index = [0, 2, 3, 4, 63, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]
    Means = np.zeros(X.shape[1])
    Stds = np.ones(X.shape[1])
    Means[index] = Mean[index]
    Stds[index] = Std[index]
    X_data = (X - Means)/Stds
    # print(X_data)
    return X_data, Means, Stds

def _cross_entropy(y_pred, Y_label):
    # compute the cross entropy
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy

def loss(y_pred, Y_label, lamda, w):
    return _cross_entropy(y_pred, Y_label) + lamda * np.sum(np.square(w))

def get_prob(X, w):
    # the probability to output 1
    return sigmoid(np.matmul(X, w))

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
    
def train_dev_split(X, y, dev_size=0.25):
    train_len = int(round(len(X)*(1-dev_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]

def accuracy(Y_pred, Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to X')
    parser.add_argument('--X_path', type=str, default='X_train', help='path to X')
    parser.add_argument('--Y_path', type=str, default='Y_train', help='path to Y')
    parser.add_argument('--nation_path', type=str, default='Nation.csv', help='path to Nation')
    opts = parser.parse_args()
    
    # X, Y = load_data(opts.X_path, opts.Y_path)
    X, Y = load_rawdata(opts.data_path, opts.X_path, opts.Y_path, opts.nation_path)
    X, Mean, Std = scaling(X)
    w = train(X, Y)
    model = np.vstack((w, Mean, Std))
    np.save("logistic_raw_17.npy",model)
    # print(model.shape)