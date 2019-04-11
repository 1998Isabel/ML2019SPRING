import sys, os
import pandas as pd
import numpy as np
import argparse
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras import losses
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import np_utils
# import matplotlib.pyplot as plt

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to X')
    parser.add_argument('--model_path', type=str)
    opts = parser.parse_args()
    model_name = opts.model_path
    
    try:
        X = np.load('X.npy')
        Y = np.load('Y.npy')
        Y = np_utils.to_categorical(Y, 7)
        # Normalizaion
        X = X / 255
    except:
        data = pd.read_csv(opts.data_path)
        X = data['feature'].str.split(' ', expand=True)
        Y = data['label'].values.astype(int)
        X = np.array(X).astype(float)
        X = X.reshape(X.shape[0], 48, 48, 1)
        np.save('X.npy',X)
        np.save('Y.npy',X)
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
    # model_name = 'Drive/ML/cnn100.h5'
    # Change data into CNN format
    X = X.reshape(X.shape[0], height, width, 1)

    # Split the data
    valid_num = 3000
    X_train, Y_train = X[:-valid_num], Y[:-valid_num]
    X_valid, Y_valid = X[-valid_num:], Y[-valid_num:]

    # Construct the model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
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

    # Callbacks
    # callbacks = []
    # modelcheckpoint = ModelCheckpoint('Drive/ML/model/weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
    # callbacks.append(modelcheckpoint)
    # csv_logger = CSVLogger('cnn_log.csv', separator=',', append=False)
    # callbacks.append(csv_logger)

    # Fit the model
    model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                                        steps_per_epoch=10*X_train.shape[0]//batch_size,
                                        epochs=epochs,
                                        validation_data=(X_valid, Y_valid))
    # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # # plt.show()
    # plt.savefig('Drive/ML/acc.png')
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('Drive/ML/loss.png')
    # # plt.show()

    # Save model
    model.save(model_name)