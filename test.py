import os
import cv2
import numpy as np
import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import matplotlib.pyplot as plt


import utils
from config import *
from sanet import SANet, load_state_dict
from dataset import NYUv2Dataset, HCDataset



def main():
    with torch.cuda.device(0): 
        torch.manual_seed(2020)
        torch.cuda.manual_seed(2020)
        np.random.seed(2020)
        torch.backends.cudnn.benchmark = True
        # network initialization
        print('Initializing model...')
        ckpt = torch.load('./pretrained/SANet-All.pth', map_location=lambda storage, loc: storage)
        net = SANet()
        load_state_dict(net.state_dict(), ckpt['model'])
        net = net.cuda()
        print('Done!')
        net.eval()
        print('evaluating...')
        # eval_dataset = NYUv2Dataset(mode='test')
        eval_dataset = HCDataset(mode='all', datalist=[15])
        eval_size = len(eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1,
                                              shuffle=False, num_workers=1)
        totalNumber = 0
        errorSum = {'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                    'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
        for i, dd in enumerate(eval_dataloader):
            img, depth, mask = dd
            img, depth, mask = img.cuda(), depth.cuda(), mask.cuda()
            with torch.no_grad():
                output = net(img)
            output = F.upsample(output, size=[depth.size(2),depth.size(3)],
                                mode='bilinear', align_corners=True)
            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = utils.evaluateError(output, depth, mask)
            errorSum = utils.addErrors(errorSum, errors, batchSize)
            averageError = utils.averageErrors(errorSum, totalNumber)

        print('NYUv2:')
        print([k + ": %.4f" % v for k,v in averageError.items()])



if __name__ == '__main__':
    main()
