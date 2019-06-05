import sys, os
import pandas as pd
import numpy as np
import argparse
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras import losses
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import np_utils
# import matplotlib.pyplot as plt

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to X')
    parser.add_argument('--weights', type=str)
    opts = parser.parse_args()
    
    try:
        print('Loading data...')
        X = np.load('Drive/ML/hw8/X.npy')
        Y = np.load('Drive/ML/hw8/Y.npy')
        Y = np_utils.to_categorical(Y, 7)
        # Normalizaion
        X = X / 255
    except:
        print('Reading data...')
        data = pd.read_csv(opts.data_path)
        X = data['feature'].str.split(' ', expand=True)
        Y = data['label'].values.astype(int)
        X = np.array(X).astype(float)
        X = X.reshape(X.shape[0], 48, 48, 1)
        np.save('Drive/ML/hw8/X.npy',X)
        np.save('Drive/ML/hw8/Y.npy',Y)
        Y = np_utils.to_categorical(Y, 7)
        # Normalizaion
        X = X / 255

    # Parameter
    height = width = 48
    num_classes = 7
    input_shape = (height, width, 1)
    batch_size = 128
    epochs = 50
    zoom_range = 0.2
    
    # Change data into CNN format
    X = X.reshape(X.shape[0], height, width, 1)

    # Split the data
    valid_num = 3000
    X_train, Y_train = X[:-valid_num], Y[:-valid_num]
    X_valid, Y_valid = X[-valid_num:], Y[-valid_num:]

    layer = [32, 64, 128, 128, 128, 128]
    st = [1, 2, 1, 2, 1, 2]
    # Construct the model
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

    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Image PreProcessing
    train_gen = ImageDataGenerator(rotation_range=25,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=[1-zoom_range, 1+zoom_range],
                                    horizontal_flip=True)
    train_gen.fit(X_train)

    # Fit the model
    # callbacks=callbacks,
    history = model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                                        steps_per_epoch=10*X_train.shape[0]//batch_size,
                                        epochs=epochs,
                                        validation_data=(X_valid, Y_valid))

    # Save weights
    weights = model.get_weights()
    for i in range(len(weights)):
        name = os.path.join(opts.weights, 'weights_np_' + str(i) + '.npy')
        np.save(name, weights[i].astype(np.float16))
    