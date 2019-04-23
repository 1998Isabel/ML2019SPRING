import os
import argparse
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from scipy.misc import imsave
from torchvision.models import vgg16, vgg19, resnet50, \
                               resnet101, densenet121, densenet169 

# using pretrain proxy model, ex. VGG16, VGG19...
model = resnet50(pretrained=True)
# or load weights from .pt file
# model = torch.load_state_dict(...)
# use eval mode
model.eval()

# loss criterion
criterion = nn.CrossEntropyLoss()
epsilon = 0.065
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# you can do some transform to the image, ex. ToTensor()
trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])


def read_data(labels_path):
    print('Loading data...')
    # try:
    #     img_list = np.load('img_list.npy')
    # except:
    #     img_list = []
    #     for i in range(200):
    #         img_path = os.path.join(input_path, str(i).zfill(3) + '.png')
    #         img = image.load_img(img_path, target_size=(224, 224))
    #         img_list.append(image.img_to_array(img))
    #     img_list = np.array(img_list)
    #     np.save('img_list', img_list)
    data = pd.read_csv(labels_path)
    labels = data['TrueLabel'].values.astype(int)
    # return img_list, labels
    return labels

# for each raw_image, target_label:
def fgsm(raw_image, target_label):
    # image = Image.fromarray(np.uint8(raw_image))
    image = Image.open(raw_image).convert('RGB')
    # image = raw_image
    
    image = trans(image)
    image = image.unsqueeze(0)
    image.requires_grad = True
    
    # set gradients to zero
    zero_gradients(image)

    output = model(image)
    target_label = torch.LongTensor([target_label])
    # print(target_label)

    loss = criterion(output, target_label)
    loss.backward() 
    
    # add epsilon to image
    sign_data_grad = torch.sign(image.grad.data)
    image = image + epsilon * sign_data_grad

    # do inverse_transform if you did some transformation
    image = image.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1))
    image = torch.clamp(image, 0, 1)
    # result = image + turbed_image
    # result = torch.clamp(result, 0.0, 1.0)
    
    image = torch.squeeze(image)
    lasttrans = transforms.ToPILImage()
    image = lasttrans(image) 

    return image
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='fgsm.py')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--part', type=str, required=True)

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    label_path = args.label
    start_part = int(args.part)
    # x, y = read_data(input_path, label_path)
    y = read_data(label_path)

    accs = []
    Ls = []
    for i in range(start_part, start_part+200):
        # print('The ', i, "'th image!")
        img_path = os.path.join(input_path, str(i).zfill(3) + '.png')
        # img = image.load_img(img_path, target_size=(224, 224))
        out_path = os.path.join(output_path, str(i).zfill(3) + '.png')
        # outimg = fgsm(x[i], y[i])
        outimg = fgsm(img_path, y[i])
        # accs.append(acc)
        # Ls.append(L)
        # now_acc = sum(accs)/len(accs)
        # now_L = sum(Ls)/len(Ls)
        # print('== acc: ', acc, 'loss: ', now_L)
        # imsave(out_path, outimg)
        outimg = imsave(out_path, outimg)