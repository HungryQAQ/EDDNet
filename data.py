import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from PIL import Image
from torchvision import transforms



class Dataset(object):
    def __init__(self, images_dir):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        label = Image.open(self.image_files[idx])
        label = self.transform(label)

        return label

    def __len__(self):
        return len(self.image_files)


class Dataset1(object):
    def __init__(self, images_dir):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        label = Image.open(self.image_files[idx])
        self.transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

        return label

    def __len__(self):
        return len(self.image_files)



