#!/usr/bin/env python
import os
import argparse
import numpy as np
from keras.models import load_model
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, DepthwiseConv2D
from keras.layers import BatchNormalization
import csv

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if len(directory) == 0:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to X')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--output_path', type=str)
    opts = parser.parse_args()

    # Parameter
    height = width = 48
    num_classes = 7
    input_shape = (height, width, 1)

    # Read the test data
    with open(opts.data_path, "r+") as f:
        line = f.read().strip().replace(',', ' ').split('\n')[1:]
        raw_data = ' '.join(line)
        length = width*height+1 #1 is for label
        data = np.array(raw_data.split()).astype('float').reshape(-1, length)
        X = data[:, 1:]
        X /= 255

    # Load weights
    w_len = 67
    weights = []
    for i in range(w_len):
        name = os.path.join(opts.weights, 'weights_np_' + str(i) + '.npy')
        ld = np.load(name).astype(np.float32)
        weights.append(ld)
    print(len(weights))

    # Load model
    layer = [32, 64, 128, 128, 128, 128]
    st = [1, 2, 1, 2, 1, 2]

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    for i in range(len(layer)):
        model.add(DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(st[i], st[i]), use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(layer[i], (1, 1), padding='same', use_bias=False, strides=(1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

    model.add(AveragePooling2D(pool_size=(2, 2),strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.set_weights(weights)

    # Plot model
    # plot_model(model,to_file='cnn_model.png')

    # Predict the test data
    X = X.reshape(X.shape[0], height, width, 1)
    ans = model.predict_classes(X)
    ans = list(ans)

    # Write prediction
    ## check the folder of out.csv is exist; otherwise, make it
    ensure_dir(opts.output_path)

    result = []
    for index, value in enumerate(ans):
        result.append("{0},{1}".format(index, value))

    with open(opts.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(len(ans)):
            # id_name = 'id_' + str(i)
            writer.writerow([i, ans[i]])