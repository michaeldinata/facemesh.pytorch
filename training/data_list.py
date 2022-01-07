import numpy as np
import random
from PIL import Image
import os
import cv2
import pandas as pd
from torchvision import transforms


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

    def __init__(self, path, phase='train'):
        self.path = path
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # self.target_transform = target_transform

        # csv_path = os.path.join(self.path, 'face_mesh_landmarks.csv')
        self.csv_file = pd.read_csv(self.path + '/face_mesh_landmarks.csv')
        self.images_path = self.path + '/images'


    def __getitem__(self, index):
        img_path = os.path.join(self.images_path,
                                self.csv_file.iloc[index, 0])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.csv_file.iloc[index, 1:]
        target = np.asarray(target)
        target = target.astype('float')
        target = target.reshape(468,3)
        min_x = int(np.min(target[:,0])-5)
        min_y = int(np.min(target[:,1])-5)
        max_x = int(np.max(target[:,0])+5)
        max_y = int(np.max(target[:,1])+5)
        img = img[min_y:max_y, min_x:max_x]
        target[:,0] = target[:,0] - min_x
        target[:,1] = target[:,1] - min_y
        img = cv2.resize(img, (192, 192))
        target[:,0] = (target[:,0]/(max_x - min_x))*192
        target[:,1] = (target[:,1]/(max_y - min_y))*192
        # target[:,0] = (target[:,0]/(max_x - min_x))
        # target[:,1] = (target[:,1]/(max_y - min_y))
        # target[:,2] = (target[:,2]/15)

        # for land in target:
        #     x,y,z = land
        #     cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1)
        # print (target)
        # cv2.imwrite("./CheckInput.jpg", img)
        # exit()

        # img = default_loader(img_path)

        # target = self.csv_file.iloc[index, 1:]
        # target = np.asarray(target)
        # target = target.astype('float')

        target = target.reshape(-1)
        img = img/127.5 - 1.0
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.csv_file)

