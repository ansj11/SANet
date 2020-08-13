import os,time
import numpy as np
import cv2
import torch
import torchvision
import torch.utils.data as data
import random
from copy import deepcopy
import scipy.io as scio
import scipy.ndimage as ndimage
from skimage import feature
from IPython import embed
from config import *
from transforms import *
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage import feature
import matplotlib.pyplot as plt
import h5py


class NYUv2Dataset(data.Dataset):
    def __init__(self, root='/home/ylab/dataset/nyu_raw_data_save/', mode='train'):
        self._mode = mode
        self._root = root
        self._paths = []
        self.load_path()
        if mode == 'train':
            self.transforms = Compose([ConvertFromInts(),
                                      Resize((IMAGE_WIDTH,IMAGE_HEIGHT)),
                                      RandomHorizontalFlip(),
                                      Normalize(),
                                      ToTensor()])
        else:
            self.transforms = Compose([ConvertFromInts(),
                                       Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                       Normalize(),
                                       ToTensor()]) 
        self._length = len(self._paths)
    
    def load_path(self):
        #input = open(self._root + 'datalist/{}_rgb.txt'.format(self._mode))
        input = open(self._root + 'datalist/{}.txt'.format(self._mode))
        alllines = input.readlines()
        for eachline in alllines:
            #path = eachline.strip()
            #self._paths.append(os.path.join(self._root, path))
            image_path, depth_path = [os.path.join('/home/ylab/dataset', i) for i in eachline.strip().split(', ')]
            self._paths.append([image_path, depth_path])

    def __getitem__(self, index):
        #path = self._paths[index]
        #rgb = cv2.imread(path)
        #depth_path = path.replace('colors', 'depth')
        image_path, depth_path = self._paths[index]
        rgb = cv2.imread(image_path)
        depth = cv2.imread(depth_path, -1)
        depth = depth.astype('float32') / 1000.
        mask = np.ones_like(depth)
        rgb, depth, mask = CropCenter((460, 620))(rgb, depth, mask) # 去除白边
        mask[depth < 0.5] = 0
        mask[depth > 10.] = 0
        yuv, depth, mask = self.transforms(rgb, depth, mask)
        sample = (yuv, depth, mask)
        return sample

    def __len__(self):
        return self._length


if __name__ == '__main__':
    mode = 'train'
    logger = SummaryWriter('./logs/')
    dataset = NYUv2Dataset(mode=mode)
    print(len(dataset))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                                   shuffle=True, num_workers=2,pin_memory=True)
    for step, (img, z, mask) in enumerate(train_dataloader):
        logger.add_image('img', make_grid(img, normalize=True, scale_each=True), i)
        logger.add_image('depth', make_grid(z, normalize=True, scale_each=True), i)
        logger.add_image('mask', make_grid(mask, normalize=True, scale_each=True), i)
        if i > 10:
            break
