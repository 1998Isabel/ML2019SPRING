import numpy as np
from skimage import color
import os, sys
import argparse
import lime
from lime import lime_image
from skimage.segmentation import slic
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import load_model
from termcolor import colored,cprint

def read_data(filename):
    # load data and model
    print(colored('Load data...', 'yellow', attrs=['bold']))
    # X = np.load('data/X.npy')
    # Y = np.load('data/Y.npy')
    try: 
        print('Using pick data...')
        X = np.load('data/X_pick.npy')
        Y = np.load('data/Y_pick.npy')
        X = X/255
        
    except:
        with open(filename, "r+") as f:
            line = f.read().strip().replace(',', ' ').split('\n')[1:]
            raw_data = ' '.join(line)
            length = width*height+1 #1 is for label
            data = np.array(raw_data.split()).astype('float').reshape(-1, length)
            X = data[:, 1:]
            Y = data[:, 0]
            # Change data into CNN format
            X = X.reshape(-1, height, width, 1)
            Y = Y.reshape(-1, 1)
            # print('Saving X.npy & Y.npy')
            img_ids = [28705, 28650, 28704, 28698, 28706, 28703, 28699]
            X = X[img_ids]
            Y = Y[img_ids]
            np.save('data/X_pick.npy', X)
            np.save('data/Y_pick.npy', Y)
    return X, Y

# two functions that lime image explainer requires
def predict(input):
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
    # TODO:
    # return ?
    input = input[:,:,:,:,0]
    return model.predict(input)

def segmentation(input):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    # TODO:
    # return ?
    # print(colored('Inside segmentation...', 'yellow', attrs=['bold']))
    return slic(input, n_segments=100, compactness=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='lime_img.py')
    parser.add_argument('--model', type=str, metavar='<#model>', required=True)
    parser.add_argument('--data', type=str, metavar='<#data>', required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    data_name = args.data
    model_name = args.model
    output_dir = args.output

    X, Y = read_data(data_name)

    # Lime needs RGB images
    # TODO:
    # x_train_rgb = ?
    # print(colored('Convert to RGB image...', 'yellow', attrs=['bold']))
    X_rgb = color.gray2rgb(X)

    print(colored('Load model...', 'yellow', attrs=['bold']))
    # model_name = 'model/cnn.h5'
    model = load_model(model_name)

    # Initiate explainer instance
    print(colored('Initiate explainer instance...', 'yellow', attrs=['bold']))
    explainer = lime_image.LimeImageExplainer()

    # img_ids = [1018, 1073, 1075, 1076, 1078, 1084, 1080]
    for idx in range(7):
        # Get the explaination of an image
        # print(colored('Get the explaination of an image...', 'yellow', attrs=['bold']))
        explaination = explainer.explain_instance(
                                    image=X_rgb[idx], 
                                    classifier_fn=predict,
                                    segmentation_fn=segmentation
                                )

        # Get processed image
        # print(colored('Get processed image...', 'yellow', attrs=['bold']))
        image, mask = explaination.get_image_and_mask(
                                        label=Y[idx],
                                        positive_only=False,
                                        hide_rest=False,
                                        num_features=5,
                                        min_weight=0.0
                                    )

        # save the image
        print('Save the image...')
        plt.imsave(os.path.join(output_dir, 'fig3_{}.jpg'.format(idx)), image.squeeze())