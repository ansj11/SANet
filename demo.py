import os, sys
import numpy as np
import argparse, time
from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from config import *
from sanet import SANet, load_state_dict
from transforms import *
import matplotlib.pyplot as plt

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='input directory',
                        default='./input', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory',
                        default='output', type=str)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=2, type=int)
     
    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default='best', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)

    args = parser.parse_args()
    return args

class Resize(object):
    def __init__(self):
        pass

    def __call__(self, image, label, mask):
        h, w, _ = image.shape
        if h >= w:
            h_out = 320
            w_out = int(w / h * 320)
        else:
            w_out = 320
            h_out = int(h / w * 320)
        image = cv2.resize(image, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w_out, w_out), interpolation=cv2.INTER_NEAREST)
        mask  = cv2.resize(mask,  (w_out, h_out), interpolation=cv2.INTER_NEAREST)
        return image, label, mask

class ImageFolder(Dataset):
    def __init__(self, dir_path, root='./imgs/'):
        filepath = []
        for root, subdir, files in os.walk(dir_path):
            for fname in files:
                if fname.endswith('jpg') or fname.endswith('png') or fname.endswith('JPG'):
                    filepath.append(os.path.join(root, fname))
        self.file_paths = filepath
        self.transforms = Compose([ConvertFromInts(),
                                   Resize(),
                                   Normalize(),
                                   ToTensor()
                                  ])
    
    def __getitem__(self, idx):
        name = self.file_paths[idx]
        image = cv2.imread(name)
        mask = np.ones(image[:,:,0].shape, dtype=np.float32)
        image, _, _ = self.transforms(image, image[:,:,0], image[:,:,0])

        return image, mask, name

    def __len__(self):
        return len(self.file_paths)

if __name__ == '__main__':
    with torch.cuda.device(DEVICE_IDS[0]):   
        args = parse_args()
        if torch.cuda.is_available() and not args.cuda:  # have cuda but no use
            print("WARNING: You might want to run with --cuda")
        folder_dataset = ImageFolder(args.input_dir, root=args.input_dir)
        input_size = len(folder_dataset)
        print("demo data size:", input_size)

        # network initialization
        print('Initializing model...')  # load pretrained resnet model
        ckpt = torch.load('./pretrained/SANet-NYUv2.pth', map_location=lambda storage, loc: storage)
        net = SANet()
        load_state_dict(net.state_dict(), ckpt['model'])
        if args.cuda:
            net = net.cuda()
        print('Done!')

        dataloader = torch.utils.data.DataLoader(folder_dataset, batch_size=1,
                                                  shuffle=False, num_workers=1)
        # setting to eval mode
        net.eval()
        print('evaluating...')
        for idx, (img, mask, name) in enumerate(dataloader):
            if args.cuda:
                img = img.cuda()
            with torch.no_grad():
                pred = net(img)
                pred_imgs = pred.cpu().numpy()
                imgs = img.cpu().numpy()
                masks = mask.cpu().numpy()
            for j in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[j].squeeze()
                mask = masks[j]
                img = imgs[j]
                h, w, _ = img.shape
                path = os.path.join(args.output_dir, name[j].split('/')[-1])
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                #np.save(path[:-3] + 'npy', pred_img)
                plt.imsave(path, pred_img, cmap='jet')
        print("Done!")


