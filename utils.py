import math
#import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
            PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_ssim(img, imclean, data_range):
    Img = img.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().clamp_(0, 1).numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        if Img.shape[1] == 1:
            SSIM += structural_similarity(np.squeeze(Iclean[i, :, :, :]), np.squeeze(Img[i, :, :, :]), 
                                          data_range=data_range, multichannel=False)
        else:
            SSIM += structural_similarity(np.squeeze(Iclean[i, :, :, :]).transpose(1, 2, 0), np.squeeze(Img[i, :, :, :]).transpose(1, 2, 0), 
                                          data_range=data_range, channel_axis=2)
    return SSIM / Img.shape[0]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

