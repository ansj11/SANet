import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import numpy as np
from IPython import embed

EPSILON = 1e-6

def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement


def evaluateError(output, target, mask):
    errors={'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
            'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    #_output, _target, nanMask, nValidElement = setNanToZero(output, target)
    mask[mask>0] = 1
    _output = output.clone()*mask
    _target = target.clone()*mask
    nValidElement = torch.sum(mask)
    nanMask = 1-mask 
    if (nValidElement.data.cpu().numpy() > 0):
        diffMatrix = torch.abs(_output - _target)
        MSE = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

        errors['RMSE'] = torch.sqrt(MSE)

        realMatrix = torch.div(diffMatrix, _target)
        realMatrix[nanMask==1] = 0
        errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

        LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
        LG10Matrix[nanMask==1] = 0
        errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

        yOverZ = torch.div(_output, _target)
        zOverY = torch.div(_target, _output)

        maxRatio = maxOfTwo(yOverZ, zOverY)

        errors['DELTA1'] = torch.sum(
            torch.le(maxRatio, 1.25).float()) / nValidElement
        errors['DELTA2'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
        errors['DELTA3'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

        errors['RMSE'] = float(errors['RMSE'].data.cpu().numpy())
        errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
        errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
        errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
        errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
        errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

    return errors


def addErrors(errorSum, errors, batchSize):
    errorSum['RMSE']=errorSum['RMSE'] + errors['RMSE'] * batchSize
    errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
    errorSum['LG10']=errorSum['LG10'] + errors['LG10'] * batchSize

    errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
    errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
    errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize

    return errorSum


def averageErrors(errorSum, N):
    averageError={'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    averageError['RMSE'] = errorSum['RMSE'] / N
    averageError['ABS_REL'] = errorSum['ABS_REL'] / N
    averageError['LG10'] = errorSum['LG10'] / N

    averageError['DELTA1'] = errorSum['DELTA1'] / N
    averageError['DELTA2'] = errorSum['DELTA2'] / N
    averageError['DELTA3'] = errorSum['DELTA3'] / N

    return averageError


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real, mask=None):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.sum(torch.abs(mask * (real - fake)) ** 2) / (torch.sum(mask) + 1e-6))
        return loss


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, fake, real, mask=None):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        diff = torch.abs(real - fake) * mask
        delta = float(self.threshold * torch.max(diff).data.cpu().numpy())
        # F.threshold(a,b,c) if a>b, then a, else a=c
        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 - delta ** 2, 0., -delta ** 2.) + delta ** 2
        part2 = part2 / (2. * delta + 1e-6)

        loss = part1 + part2
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-6)
        return loss


class GradLoss2(nn.Module):
    def __init__(self):
        super(GradLoss2, self).__init__()

    def forward(self, fake, real, mask=None):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        real = real + (mask == 0).float() * fake
        scales = [1, 2, 4, 8, 16]
        grad_loss = 0
        for scale in scales:
            pre_dx, pre_dy, pre_m_dx, pre_m_dy = gradient2(fake, mask, scale)
            gt_dx, gt_dy, gt_m_dx, gt_m_dy = gradient2(real, mask, scale)
            diff_x = pre_dx - gt_dx
            diff_y = pre_dy - gt_dy
            grad_loss += torch.sum(torch.abs(diff_x*pre_m_dx))/(torch.sum(pre_m_dx) + 1e-6) + torch.sum(torch.abs(diff_y*pre_m_dy))/(torch.sum(pre_m_dy) + 1e-6)

        return grad_loss

def gradient(depth, mask):
    D_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    D_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    return D_dx, D_dy, mask_dx, mask_dy


class NormalLoss(nn.Module):
    """
    compute normal vector loss: 
    loss = sum(1-fake'*real/|fake|/|real|)
    """
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, fake, real, mask=None):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        pre_dx, pre_dy, pre_m_dx, pre_m_dy = gradient(fake, mask)
        gt_dx, gt_dy, gt_m_dx, gt_m_dy = gradient(real, mask)
        inner_dx = (pre_dx * gt_dx)[:,:,:-1,:]
        inner_dy = (pre_dy * gt_dy)[:,:,:,:-1]
        pred_dxx = (pre_dx**2)[:,:,:-1,:]
        pred_dyy = (pre_dy**2)[:,:,:,:-1]

        gt_dxx = (gt_dx**2)[:,:,:-1,:]
        gt_dyy = (gt_dy**2)[:,:,:,:-1]

        loss = 1- torch.div(inner_dx+inner_dy+1,
                            torch.sqrt(pred_dxx+pred_dyy+1)*torch.sqrt(gt_dxx+gt_dyy+1))
        mask = gt_m_dx[:,:,:-1,:] * gt_m_dy[:,:,:,:-1]
        return torch.sum(loss*mask) / (torch.sum(mask) + 1e-6)


class MREL(nn.Module):
    def __init__(self):
        super(MREL, self).__init__()

    def forward(self, fake, real, mask=None):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        diff = torch.abs(fake-real)*mask
        loss = torch.sum(mask*(diff/(real + 1e-6)))/(torch.sum(mask) + 1e-6)
        return loss


# global focal relative loss
class SRLL(nn.Module):
    def __init__(self, use_gpu=True):
        super(SRLL, self).__init__()
        self.use_gpu = use_gpu
        self.ids = 0

    def get_random_ids(self):
        l = []
        for row in range(16):
            for col in range(16):
                y = np.random.randint(row*15, row*15+15)
                x = np.random.randint(col*20, col*20+20)
                l.append(320*y+x)
        return l

    def forward(self, fake, real, mask):
        N, _, H, W = real.size()
        if not fake.shape == real.shape:
            fake =  F.upsample(fake, size=(H, W), mode='bilinear')
        self.ids = self.get_random_ids()
        fake_i = fake.view(N, H * W)[:, self.ids]
        real_i = real.view(N, H * W)[:, self.ids]
        mask_i = mask.view(N, H * W)[:, self.ids]
        N, HW = fake_i.size()
        fake_j = fake_i.repeat(1, HW).reshape(N, -1, HW)
        mask_j = mask_i.repeat(1, HW).reshape(N, -1, HW)
        real_j = real_i.repeat(1, HW).reshape(N, -1, HW)
        real_i = real_j.permute(0, 2, 1)
        fake_diff = fake_j.permute(0, 2, 1) - fake_j
        mask_diff = mask_j.permute(0, 2, 1) * mask_j
        #real_diff = real_i.reshape(N, -1, 1) - real_j
        
        r1 = (real_i > 1.02 * real_j).float()
        r2 = (real_j > 1.02 * real_i).float()
        r0 = ((real_i <=1.02 * real_j) * (real_j <=1.02 * real_i)).float()
        loss =  torch.log(1 + torch.exp(-fake_diff)) * r1 +  torch.log(1 + torch.exp(fake_diff)) * r2 + fake_diff**2 * r0
        ret  = torch.sum(loss * mask_diff * (1-torch.sigmoid((r1-r2)*fake_diff))**2) / (torch.sum(mask_diff) + 1e-6)
        return ret



