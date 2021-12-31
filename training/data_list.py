import numpy as np
import random
from PIL import Image
import os
import cv2
import pandas as pd


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
        self.target_transform = target_transform
        self.loader = loader

        # csv_path = os.path.join(self.path, 'face_mesh_landmarks.csv')
        self.csv_file = pd.read_csv(self.path + 'face_mesh_landmarks.csv')
        self.images_path = self.path + '/images'


    def __getitem__(self, index):
        img_path = os.path.join(self.images_path,
                                self.csv_file.iloc[index, 0])
        img = default_loader(img_path)

        target = self.csv_file.iloc[index, 1:]
        target = np.asarray(target)
        target = target.astype('float').reshape(-1, 3)

        return img, target

    def __len__(self):
        return len(self.csv_file)

