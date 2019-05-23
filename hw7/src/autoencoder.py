import os
import argparse
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Reshape
from keras.models import Model
import keras.backend as K
import PIL
import matplotlib.pyplot as plt

def read_data(input_path, img_path):
    # try:
    img_list = np.load(img_path)
    img_list = img_list/255.0
    print('load saved img_list')
    # except:
    #     img_list = []
    #     for i in range(1,40001):
    #         img_path = os.path.join(input_path, str(i).zfill(6)+".jpg")
    #         img = image.load_img(img_path)
    #         img_list.append(image.img_to_array(img))
    #     img_list = np.array(img_list)
    #     img_list = np.reshape(img_list, (40000,3,32,32))
    #     img_list = img_list/255.0
    #     print()
    #     np.save(img_path, img_list)
    #     print('save img_list')
    print(img_list.shape)
    return img_list

def train(train_data, all_model):
    train_data = np.reshape(train_data, (40000,-1))
    # train_data = np.reshape(train_data, (40000,32,32,3))
    print(train_data.shape) #(40000, 3072)
    input_dim = train_data.shape[1]
    # input_dim = (32,32,3)
    encoding_dim = 128
    autoencoder = Sequential()

    # Encoder Layers
    autoencoder.add(Dense(3072, input_shape=(input_dim,), activation='relu'))
    # autoencoder.add(Dropout(0.4))
    autoencoder.add(Dense(1024, activation='relu'))
    # autoencoder.add(Dropout(0.4))
    autoencoder.add(Dense(512, activation='relu'))
    # autoencoder.add(Dropout(0.4))
    autoencoder.add(Dense(256, activation='relu'))

    # Decoder Layers
    autoencoder.add(Dense(512, activation='relu'))
    # autoencoder.add(Dropout(0.4))
    autoencoder.add(Dense(1024, activation='relu'))
    # autoencoder.add(Dropout(0.4))
    autoencoder.add(Dense(3072, activation='relu'))
    # autoencoder.add(Dropout(0.4))
    # autoencoder.add(Dense(3, activation='sigmoid'))
    # autoencoder.add(Reshape(input_dim))

    autoencoder.summary()

    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(train_data, train_data, epochs=100, batch_size=256)
    
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('Drive/ML/hw7/plot/loss_dnn_5.png')
    # plt.show()

    autoencoder.save(all_model)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='train.csv')
    parser.add_argument('--imglist', type=str, default='train.csv')
    parser.add_argument('--all', type=str, default='train.csv')
    opts = parser.parse_args()
    imgs = read_data(opts.input, opts.imglist)
    train(imgs, opts.all)
    