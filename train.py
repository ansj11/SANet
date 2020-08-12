import os, sys
import numpy as np
import argparse, time
import matplotlib, cv2
from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

from config import *
from dataset import HCDataset
from sanet import SANet
from utils import *
from logger import Logger

torch.backends.cudnn.benchmark = True

# loss
rmse = RMSE()
berhu = BerHu()
grad = GradLoss2()
norm = NormalLoss()
mrel = MREL()
srfl = SRLL()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=NUM_EPOCHS, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--bs', dest='bs',
                        help='batch_size',
                        default=BATCH_SIZE, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=NUM_WORKER, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='display interval',
                        default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory',
                        default='models', type=str)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=LEARNING_RATE, type=float)
    parser.add_argument('--decay_step', dest='decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=10, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--start_at', dest='start_epoch',
                        help='epoch to start with',
                        default=0, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default='latest', type=str)
    parser.add_argument('--device', dest='device',
                        help='factor of regularization loss',
                        default=0., type=float)

    args = parser.parse_args()
    return args

args = parse_args()

def main():
    with torch.cuda.device(DEVICE_IDS[0]):
        cv2.setNumThreads(0)
        #torch.manual_seed(2019)
        torch.backends.cudnn.benchmark = True
            
        output_dir = os.path.join('/home/ylab/models/',os.path.basename(os.getcwd()))
        if not os.path.exists(output_dir):  # ouput dir
            os.makedirs(output_dir)
        if not os.path.exists(args.output_dir):
            os.symlink(output_dir,args.output_dir)
    
        log_dir = os.path.join('/home/ylab/logs/',os.path.basename(os.getcwd()))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists('./logs'):
            os.symlink(log_dir, './logs')
        train_logger = Logger('./logs/train')
        valid_logger = Logger('./logs/valid')

        train_dataset = HCDataset(mode='train')
        train_size = len(train_dataset)
        print("train data size:", train_size)
        train_sample_weight = torch.from_numpy(np.array(train_dataset._sample_weights))
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weight.type('torch.DoubleTensor'), len(train_sample_weight))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                                    shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        eval_dataset = HCDataset(mode='test', datalist=[14])
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)

        net = _model_init()
        # optimizer
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == "rms":
            optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)

        train_per_epoch = int(train_size/args.bs)
        COUNT, BEST_STEP, MIN_LOSS = args.start_epoch*train_per_epoch, 0, np.inf
        #scheduler = get_scheduler(optimizer, policy='multi-step')
        print("Start training...")
        start_eval = 0
        epoch_time = time.time()
        start_time = epoch_time
        for epoch in range(args.start_epoch, args.max_epochs):
            #scheduler.step()
            for step, batch in enumerate(train_dataloader):
                img, z, mask = batch
                if args.cuda:
                    img, z, mask = img.cuda(), z.cuda(), mask.cuda()
                COUNT += 1
                if COUNT % 1000 == 0 or start_eval==0:
                    BEST_STEP, MIN_LOSS = test(net, eval_dataloader, epoch, COUNT, valid_logger, BEST_STEP, MIN_LOSS)
                    # scheduler.step(MIN_LOSS)
                    print('leaning rate = %.7f, elapse time = %.f' % (optimizer.param_groups[0]['lr'], time.time() - epoch_time))
                    torch.cuda.empty_cache()
                start_eval += 1
                train(net, epoch, COUNT, img, z, mask, optimizer, train_logger)
            cur_time = time.time()
            print('Time for current epoch: {:f}'.format(cur_time - epoch_time))
            print('Total time used: {:f}'.format(cur_time - start_time))
        print('Done!')

def test(net, eval_dataloader, epoch, COUNT, logger, BEST_STEP, MIN_LOSS):
    torch.set_grad_enabled(False)
    net.eval()
    print('validate dataset: ', len(eval_dataloader))
    rmse_loss, berhu_loss, grad_loss = 0, 0, 0
    norm_loss, mrel_loss, srfl_loss = 0, 0, 0
    for idx, batch in enumerate(eval_dataloader):
        img, z, mask = batch
        if args.cuda:
            img, z, mask = img.cuda(), z.cuda(), mask.cuda()
            img = F.upsample(img, size=(240, 320), mode='bilinear')
            pred = net(img)
            rmse_loss += rmse(pred, z, mask).item()
            berhu_loss += berhu(pred, z, mask).item()
            grad_loss += grad(pred, z, mask).item()
            norm_loss += norm(pred, z, mask).item()
            mrel_loss += mrel(pred, z, mask).item()
            srfl_loss += srfl(pred, z, mask).item()
    vol = idx + 1
    rmse_loss, berhu_loss, grad_loss = rmse_loss/vol, berhu_loss/vol, grad_loss/vol
    norm_loss, mrel_loss, srfl_loss = norm_loss/vol, mrel_loss/vol, srfl_loss/vol
    loss = berhu_loss + grad_loss #+ norm_loss #+ srfl_loss
    info = {'loss': loss, 'rmse': rmse_loss, 'berhu': berhu_loss, 'grad': grad_loss,
            'norm': norm_loss, 'mrel': mrel_loss, 'RL': srfl_loss}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, COUNT)
    info = {'depths': z[:4].squeeze(1).cpu().numpy(), 'd-pred': pred[:4].squeeze(1).cpu().numpy(),
            'masks': mask[:4].squeeze(1).cpu().numpy(), 'images': img[:4].cpu().numpy().transpose([0,2,3,1])}
    for tag, images in info.items():
        logger.image_summary(tag, images, COUNT)
    if loss < MIN_LOSS:
        BEST_STEP, MIN_LOSS = COUNT, loss
        save_name = os.path.join(args.output_dir, 'model-best.model')
        torch.save(net, save_name)
        torch.save({'epoch': epoch + 1,
                    'model': net.state_dict()}, save_name[:-5] + 'pth')
    print("[epoch %2d][iter %5d]loss: %.4f RMSE loss: %.4f berhu loss: %.4f  Grad: %.4f Norm: %.4f MREL: %.4f RL: %.4f Best: [%d, %.4f]" \
          % (epoch, COUNT,loss, rmse_loss, berhu_loss, grad_loss, norm_loss, mrel_loss, srfl_loss, BEST_STEP, MIN_LOSS))
    with open('val.txt', 'a') as f:
        f.write(
            "[epoch %2d][iter %5d] loss: %.4f RMSE loss: %.4f berhu loss: %.4f  Grad: %.4f Norm: %.4f MREL: %.4f RL: %.4f Best: [%d, %.4f]\n" \
            % (epoch, COUNT,loss, rmse_loss, berhu_loss, grad_loss, norm_loss, mrel_loss, srfl_loss, BEST_STEP, MIN_LOSS))
    return BEST_STEP, MIN_LOSS

def train(net, epoch, COUNT, img, z, mask, optimizer, logger):
    torch.set_grad_enabled(True)
    net.train()
    optimizer.zero_grad()
    img = F.upsample(img, size=(240, 320), mode='bilinear')
    pred = net(img)
    rmse_loss = rmse(pred, z, mask)
    berhu_loss = berhu(pred, z, mask)
    grad_loss = grad(pred, z, mask)
    norm_loss = norm(pred, z, mask)
    mrel_loss = mrel(pred, z, mask)
    srfl_loss = srfl(pred, z, mask)
    loss = berhu_loss + grad_loss * (epoch > 15) + norm_loss * (epoch > 30) *10 + srfl_loss * 0.5 *(epoch > 30)
    loss.backward()
    #updateBN(net)
    optimizer.step()
    if COUNT % args.disp_interval == 0:
        print("[epoch %2d][iter %4d] loss: %.4f RMSE: %.4f berhu: %.4f Grad: %.4f Norm: %.4f MREL: %.4f RL: %.4f" % (
            epoch, COUNT, loss.item(), rmse_loss.item(), berhu_loss.item(), grad_loss.item(), norm_loss.item(), mrel_loss.item(), srfl_loss.item()))
    if COUNT % 1000 == 0:
        info = {'loss': loss.item(), 'rmse': rmse_loss.item(), 'berhu': berhu_loss.item(), 'grad': grad_loss.item(),
                'norm': norm_loss.item(), 'mrel': mrel_loss.item(), 'RL': srfl_loss.item()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, COUNT)
        info = {'depths': z[:4].squeeze(1).cpu().numpy(), 'd-pred': pred[:4].squeeze(1).detach().cpu().numpy(),
                'masks': mask[:4].squeeze(1).cpu().numpy(), 'images': img[:4].cpu().numpy().transpose([0,2,3,1])}
        for tag, value in info.items():
            logger.image_summary(tag, value, COUNT)
    if COUNT % 1000 == 0:
        # save model
        save_name = os.path.join(args.output_dir, 'model-latest.model')
        torch.save(net, save_name)
        print('save model: {}'.format(save_name))

def updateBN(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(L1_COEF * torch.sign(m.weight.data))    # L1 regularizer

def _model_init():
    # network initialization or resume
    if args.resume:
        load_name = os.path.join(args.output_dir, 'model-{}'.format(args.checkepoch))
        print("loading model %s" % (load_name))
        if load_name.endswith('pth'):
            net = Net()
            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            load_state_dict(net.state_dict(), checkpoint['model'])
        else:
            net = torch.load(load_name, map_location=lambda storage, loc: storage)
    else:
        print('Initializing model...')
        net = Net()
    if args.cuda:
        net = net.cuda(DEVICE_IDS[0])
    if len(DEVICE_IDS) > 1:
        print('DataParallel!')
        net = nn.DataParallel(net, device_ids=DEVICE_IDS, output_device=DEVICE_IDS[0])
    return net

def adjust_learning_rate(optimizer, step):
    decay_rate = 0.96
    if step % args.decay_step ==0 and step > 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate # exponential decay when step > 100

def get_scheduler(optimizer, policy):
    if policy == 'lambda':
        def lambda_rule(step):
            lr = 1.0 - max(0, step - 1000) / float(MAX_STEP + 1)
            return lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.00001, patience=10)
    elif policy == 'multi-step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, MILESTONES, gamma=0.2)
    elif policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler

def load_state_dict(own_state,  new_state_dict):
    for name, param in new_state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                continue
                #raise RuntimeError('While copying the parameter named {}, '
                #        'whose dimensions in the model are {} and '
                #        'whose dimensions in the checkpoint are {}.'
                #        .format(name, own_state[name].size(), param.size()))

if __name__ == '__main__':
    main()
