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


paths = {#'2D-3D/xyz/': 0, 
         'suncg_new/': 1, 
         'scannet_parse/': 2, 
         #'nyu_raw_data_save/': 3,
         'CAD/': 4, 
         #'PrincetonTrackingPeople/': 5, 
         'URFallDetectionPeople/': 6,
         'fall_data/': 6,
         'cvpr/': 7,
         'mypeople/': 8,
         #'MegaDepth/': 9,
         #'KITTI/': 10,
         #'RealSense_outdoor/': 11,
         #'RealSense_indoor': 12,
         #'data_new/': 13,
         'nyu_new/': 14,
         #'mannequinchallenge/': 15,
         'RealSense/': 16
        }
datalist = list(paths.values()) 

class HCDataset(data.Dataset):
    def __init__(self, root='/home/ylab/dataset/', datalist=datalist, mode='train'):
        self._root = root
        self._mode = mode
        self._datalist = datalist
        self._paths = []
        self._tags = []
        self._nums = [0 for i in range(max(datalist) + 1)]
        self.load_path()
        self.cal_weights()
        if mode == 'train':
            self.transforms = Compose([ConvertFromInts(),
                                      Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                      RandomHorizontalFlip(),
                                      Normalize(),
                                      ToTensor()])
        else:
            self.transforms = Compose([ConvertFromInts(),
                                       Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                       Normalize(),
                                       ToTensor()])

    def load_path(self):
        input = open('data/{}.txt'.format(self._mode))
        alllines = input.readlines()
        for eachline in alllines:
            path, tag = eachline.strip().split()
            tag = int(tag)
            if tag in self._datalist:
                self._paths.append(os.path.join(self._root, path))
                self._tags.append(tag)
                self._nums[tag] += 1

    def cal_weights(self):
        total = sum(self._nums)
        self._weights = deepcopy(self._nums)
        for i in range(len(self._nums)):
            if self._nums[i] == 0:
                self._weights[i] = 0
            elif i == 1:
                self._weights[i] = total / self._nums[i] * 0.6
            elif i in [3, 4, 6, 7, 8]:
                self._weights[i] = total / self._nums[i] 
            else:
                self._weights[i] = total / self._nums[i]
        self._sample_weights = deepcopy(self._tags)
        for i in range(len(self._tags)):
            self._sample_weights[i] = self._weights[self._tags[i]]
    
    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        path = self._paths[index]
        tag = self._tags[index]
        rgb = cv2.imread(path) if not path.endswith('h5') else None
        if tag in [0, 3, 4, 5, 7] or (tag == 6 and 'URFallDetectionPeople' in path):
            depth_path = path.replace('rgb', 'depth')
        elif tag == 1:
            if 'mlt' in path:
                depth_path = path.replace('rgb', 'depth').replace('mlt', 'depth')
            else:
                depth_path = path.replace('rgb', 'depth')
        elif tag == 2:
            depth_path = path[:-3].replace('rgb', 'depth') + 'png'
        elif tag == 6 and 'fall_data' in path:
            depth_path = path.replace('rgb', 'd')
        elif tag == 8:
            depth_path = path.replace('color/color_', 'mapped_depth/mapped_depth')
        elif tag == 9:
            depth_path = path[:-3].replace('imgs', 'depths') + 'h5'
        elif tag == 10:
            depth_path = path.replace('image', 'depth')
        elif tag in [11, 12, 13, 16]:
            depth_path = path.replace('color/color', 'depth/depth')
        elif tag == 14:
            depth_path = path.replace('jpg', 'png') if self._mode == 'train' else path.replace('colors', 'depth')
        elif tag == 15:
            pass
        if tag == 9:
            with h5py.File(depth_path, 'r') as f:
                depth = f.get('/depth')[:]
        elif tag == 15:
            with h5py.File(path, 'r') as f:
                rgb = np.array(f.get('/gt/img_1')).astype('float32') * 255.0
                depth = np.array(f.get('/gt/gt_depth')).astype('float32')
        else:
            depth = cv2.imread(depth_path, -1)
        mask = np.ones(depth.shape, dtype=np.float32)
        if tag == 0:
            depth = depth.astype('float32') / 512.
            mask[depth>120] = 0
            depth = depth * mask
            rgb, depth, mask = CropCenter((810, 1080))(rgb, depth, mask)
        elif tag in [3, 7]:
            depth = depth.astype('float32') / 1000
            rgb, depth, mask = CropCenter((460, 620))(rgb, depth, mask)
        elif tag == 4:
            if depth.dtype == 'uint8':
                depth = depth.astype('float32') / 255 * 20
            elif depth.dtype == 'uint16':
                depth = depth.astype('float32') / 1000
        elif tag == 6:
            if 'URFallDetectionPeople' in path:
                depth = depth.astype('float32') / 65535 * 7
            elif 'fall_data' in path:
                depth = depth.astype('float32') / 65535 * 6
        elif tag == 9:
            pass
        elif tag == 10:
            depth = depth.astype('float32') / 256.0
            depth = depth / 8.
            if self._mode == 'train':
                rgb, depth, mask = RandomCrop((360, 480))(rgb, depth, mask)
            else:
                rgb, depth, mask = CropCenter((352, 480))(rgb, depth, mask)
        elif tag == 14:
            if self._mode == 'train':
                depth = depth.astype('float32') / 255 * 10
            else:
                depth = depth.astype('float32') / 1000
            rgb, depth, mask = CropCenter((460, 620))(rgb, depth, mask)
        elif tag == 15:
            mask = (depth > 1e-8).astype('float32')
        else:
            depth = depth.astype('float32') / 1000
        mask[depth < 0.5] = 0
        mask[depth > 10.] = 0       
        if self._mode == 'train' and tag == 7:
            rgb, depth, mask = Resize((IMAGE_WIDTH, IMAGE_HEIGHT))(rgb, depth, mask)
            mask_contour = feature.canny(depth, sigma=3)
            mask_contour = (4 * mask_contour).astype('float32')
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(mask_contour, kernel, iterations=1)
            dilation += 1
            #plt.imsave('./edges/%d-edge.png'%index, dilation, cmap='jet', format='png')
            #plt.imsave('./edges/%d-imgs.png'%index, rgb[:,:,::-1], format='png')
            #plt.imsave('./edges/%d-dept.png'%index, depth, cmap='jet', format='png')
            mask = mask * dilation
        yuv, depth, mask = self.transforms(rgb, depth, mask)
        sample = (yuv, depth, mask)
        return sample



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
