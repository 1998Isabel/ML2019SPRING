import os
import argparse
import numpy as np
from skimage.io import imread, imsave
# from PIL import Image

def read_data(input_path):
    # try:
    #     img_path = os.path.join(input_path, 'raw_imglist_new.npy')
    #     img_list = np.load(img_path)
    #     print(img_list.shape)
    # except:
    img_list = []
    print('Start reading images...')
    for i in range(415):
        # data_path = os.path.join(input_path, 'Aberdeen')
        img_path = os.path.join(input_path, str(i)+".jpg")
        img = imread(img_path)
        img_list.append(img.flatten())
    img_list = np.array(img_list).astype('float32')
    # np.save('raw_imglist_new.npy', img_list)
    print(img_list.shape)
    # print('save img list')
    # img_list = np.array(img_list).astype('float32')
    # img_list = np.reshape(img_list, (415,-1))
    # print(img_list.shape)
    return img_list

def process(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def zeroMean(dataMat, eigface_path):        
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    
    mean_path = os.path.join(eigface_path, 'mean_face.jpg') 
    average = process(meanVal)
    imsave(mean_path, average.reshape((600,600,3)))
    return newData, meanVal

def pca_svd(imgs, output_path, pick_path):
    # newData, meanVal = zeroMean(imgs, eigface_path) 
    meanVal = np.mean(imgs, axis = 0) 
    newData = imgs - meanVal
    print('Start svd...')
    u, d, v = np.linalg.svd(newData.T, full_matrices = False)
    print('eigVals shape: ', d.shape)
    print('eigVects shape: ', u.shape)
    Index = 5
    # pick = [0, 68, 112, 117, 167]
    pick = [0]
    for t in pick:
        # picked_img = imread(os.path.join(pick_path, str(t)+'.jpg'))
        picked_img = imread(pick_path)  
        X = picked_img.flatten().astype('float32') 
        X -= meanVal
        print('reconsturcting ', t)
        picked_faces = u.T[:Index]
        weight = X.dot(picked_faces.T)
        
        # Reconstruction
        reconstruct = process(weight.dot(u.T[:Index]) + meanVal)
        # print(reconstruct.shape)
        # out_path = os.path.join(output_path, str(t)+"_reconstruct.jpg")
        imsave(output_path, reconstruct.reshape((600,600,3)))

    # print('Saving eigenfaces...')
    # for i in range(Index):
    #     print('eigenface ', i)
    #     eigenface = process(u.T[i])
    #     eigen_path = os.path.join(eigface_path, str(i)+"_eigen.jpg")
    #     imsave(eigen_path, eigenface.reshape((600,600,3)))
    
    print('The 5 eigen values:')
    for j in range(5):
        number = d[j] * 100 / sum(d)
        print(j, number)
    return


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--pick', type=str)
    opts = parser.parse_args()
    imgs = read_data(opts.input)
    pca_svd(imgs, opts.output, opts.pick)