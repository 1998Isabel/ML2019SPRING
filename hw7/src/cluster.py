import os
import argparse
import csv
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_img(input_path):
    img_list = []
    for i in range(1,40001):
        img_path = os.path.join(input_path, str(i).zfill(6)+".jpg")
        img = image.load_img(img_path)
        img_list.append(image.img_to_array(img))
    img_list = np.array(img_list)
    img_list = np.reshape(img_list, (40000,3,32,32))
    img_list = img_list/255.0
    return img_list

def encode(img_list, allmodel):
    dims = 256
    img_list = np.reshape(img_list, (40000,-1))
    model = load_model(allmodel)
    # img_encode = encoder.predict(img_list)
    encoder = K.function([model.layers[0].input], [model.layers[3].output])
    img_encode = encoder([img_list])[0].reshape(-1,dims)
    # auto_encode = np.array(img_encode)
    # np.save('Drive/ML/hw7/feature/auto_encode_dnn5.npy', auto_encode)
    print(img_encode.shape)
    return img_encode

def dopca(data):
    reduced_dim = 100
    # pca = PCA(n_components=100, whiten=True)
    # pca.fit(data)
    # data_pca = pca.transform(data)
    pca = PCA(n_components=reduced_dim, copy=False, whiten=True, svd_solver='full')
    data = pca.fit_transform(data)
    return data

def read_test(test_path):
    test = pd.read_csv(test_path)
    test1 = test['image1_name'].values
    test2 = test['image2_name'].values
    test1 = np.array(test1)
    test2 = np.array(test2)
    print('test shape: ', test1.shape, test2.shape)
    return test1, test2

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='train.csv')
    parser.add_argument('--input', type=str, default='train.csv')
    parser.add_argument('--encoder', type=str, default='train.csv')
    parser.add_argument('--output', type=str, default='train.csv')
    opts = parser.parse_args()
    imgs = load_img(opts.input)
    img_encode = encode(imgs, opts.encoder)

    # PCA
    print("Start PCA")
    img_encode = dopca(img_encode)
    # pca_encode = np.array(img_encode)
    # np.save('Drive/ML/hw7/feature/pca_encode_26.npy', pca_encode)
    print('after pca shape: ', img_encode.shape)

    # km = KMeans(n_clusters=2)  #K=2群
    # y_pred = km.fit_predict(img_encode)
    n_iter = 300
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=n_iter).fit(img_encode)
    y_pred = kmeans.labels_
    print('kmeans shape: ', y_pred.shape)

    # # plotting
    # encoded_imgs = encoder.predict(x_test)
    # plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
    # plt.colorbar()

    test1, test2 = read_test(opts.test)
    
    with open(opts.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(test1.shape[0]):
            # writer.writerow([id_name, float(Y[i])])
            if y_pred[int(test1[i])-1] == y_pred[int(test2[i])-1]:
                writer.writerow([str(i), str(1)])
            else:
                writer.writerow([str(i), str(0)])
