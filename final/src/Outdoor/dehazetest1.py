from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from misc import *
import dehaze1113 as net

w, h = 1536 * 4, 1536

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=1024, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=1024, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='./src/Outdoor/sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--directory', default='', help="path to output result")
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = 61677
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

netG=net.Dense_rain_cvprw3()

g = torch.load(opt.netG)
new_g = dict()
for (k1, v1) in g.items():
  if 'conv.1' in k1:
      new_g[k1.replace('conv.1', 'conv1')] = v1
  elif 'norm.1' in k1:
      new_g[k1.replace('norm.1', 'norm1')] = v1
  elif 'norm.2' in k1:
      new_g[k1.replace('norm.2', 'norm2')] = v1
  elif 'conv.2' in k1:
      new_g[k1.replace('conv.2', 'conv2')] = v1
  else:
      new_g[k1] = v1

if opt.netG != '':
  netG.load_state_dict(new_g)

netG.train()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)

val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
label_d = torch.FloatTensor(opt.batchSize)


target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
depth = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
ato = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)


val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_depth = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_ato = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netG.cuda()

with torch.no_grad():
  target, input, depth, ato = target.cuda(), input.cuda(), depth.cuda(), ato.cuda()
  val_target, val_input, val_depth, val_ato = val_target.cuda(), val_input.cuda(), val_depth.cuda(), val_ato.cuda()

label_d = Variable(label_d.cuda())

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())
import time

ganIterations = 0
for epoch in range(1):
  heavy, medium, light=200, 200, 200
  for i, data in enumerate(valDataloader, 0):
    with torch.no_grad():
      if 1:
        t0 = time.time()

        data_val = data

        val_input_cpu, val_target_cpu, path = data_val

        val_target_cpu, val_input_cpu = val_target_cpu.float().cuda(), val_input_cpu.float().cuda()
        val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)

        val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
        val_target = val_target_cpu


        z=0
        label_cpu = torch.FloatTensor(opt.batchSize).fill_(z)


        label_cpu2=0

        label_d.data.fill_(label_cpu2)

        label_cpu = label_cpu.long().cuda()
        label_cpu = Variable(label_cpu)


        for idx in range(val_input.size(0)):
          single_img = val_input[idx,:,:,:].unsqueeze(0)
          val_inputv = single_img

          x_hat_val = netG(val_target)

        from PIL import Image
        resukt = torch.cat([val_inputv,x_hat_val],3)
        tensor = x_hat_val.data.cpu()

        from PIL import Image

        directory = opt.directory
        if not os.path.exists(directory):
            os.makedirs(directory)


        name=''.join(path)
        filename = os.path.join(directory, str(i)+'.png')

        tensor = torch.squeeze(tensor)
        tensor=norm_range(tensor, None)
        print('Patch:'+str(i))


        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        im.save(filename)
        t1 = time.time()
        print('running time:'+str(t1-t0))
