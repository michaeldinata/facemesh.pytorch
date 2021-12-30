import numpy as np
import random
from PIL import Image
import os
import cv2


def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

class ImageList(object):

    def __init__(self, path, phase='train', transform=None, target_transform=None,
                 loader=default_loader):
        self.path = path
        self.phase = phase
        self.transform = transform
        self.target_transform = transform
        self.loader = loader


    def __getitem__(self, index):

        return img, target

    def __len__(self):
        # return len

