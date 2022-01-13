import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
import itertools

import cv2
import config
import network
import pre_process as prep
import lr_schedule
from util import *
from data_list import ImageList

sys.path.append('../ThermalFaceDetection')
import ThermalFaceDetection

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

def GetDatasets(config):
    apply_cropping = config.apply_cropping
    dsets = {}
    dsets['train'] = ImageList(path=config.train_path_prefix)
    dsets['eval'] = ImageList(path=config.eval_path_prefix, phase='eval')
    dsets['test'] = ImageList(path=config.test_path_prefix, phase='test')
    return dsets

def GetDataLoaders(config, dsets):
    dset_loaders = {}
    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)
    dset_loaders['eval'] = util_data.DataLoader(dsets['eval'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)
    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.test_batch_size,
                                                shuffle=False, num_workers=config.num_workers)
    return dset_loaders

def InitializeModel(config):
    model = network.FaceMesh()
    return model

def main():
    fd = ThermalFaceDetection.FaceDetection()
    test_path = config.test_path_prefix
    video_path = test_path + '2pj4jwlz.mkf_TH.mp4'
    cap = cv2.VideoCapture(video_path)

    MakeDir("./InferenceResults/")
    use_gpu = torch.cuda.is_available()
    # print ("Preparing Data!")
    # dsets = GetDatasets(config)
    # dset_loaders = GetDataLoaders(config, dsets)
    # print ("Done! Preparing Data")
    
    print ("Getting Networks!!")
    model = InitializeModel(config)
    model.load_state_dict(torch.load('./Snapshots/10.pth'))

    if use_gpu:
        model = model.cuda()
    
    model.eval()

    while cap.isOpened():
        success, image = cap.read()
        print(success)
        cv2.imshow('RGB MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rgb_image = cv2.resize(rgb_image, (192, 192))
        dets = fd.run(image)

        if dets is None:
            print("None")
            continue

        dets = dets[0]

        ### Retrieve the respective coordinates of facial features ###
        bounding_box = np.round_(dets[:4]).astype(np.uint32)
        # prediction_acc = dets[4]
        left_eye = np.round_(dets[5:7]).astype(np.uint32)
        right_eye = np.round_(dets[7:9]).astype(np.uint32)
        nose = np.round_(dets[9:11]).astype(np.uint32)
        left_mouth = np.round_(dets[11:13]).astype(np.uint32)
        right_mouth = np.round_(dets[13:15]).astype(np.uint32)

        cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 0, 0), 2)

        cv2.imwrite("./InferenceResults/iter.jpg", image)
        print ("Done!")

        exit()

    exit()
    for i, batch in enumerate(dset_loaders['test']):
        input, target = batch

        if use_gpu:
            input = input.float().cuda()
            target = target.float().cuda()
        else:
            target = target.float()

        pred = model(input)
        detections, confidences = pred
        # print(detections)
        print (input.size())
        print (target.size())

        input1 = (input.clone() + 1.0) * 127.5
        input1 = input1.permute(0,2,3,1)
        number_of_images = input1.size(0)
        for n in range(number_of_images):
            inp_target = input1[n,:,:,:].clone().cpu().numpy()
            inp_pred = input1[n,:,:,:].clone().cpu().numpy()
            targ = target[n,:].cpu().numpy().reshape(468,3)
            det = detections[n,:].detach().cpu().numpy().reshape(468,3)

            # inp_target = np.array(inp_target)
            inp_target = np.ascontiguousarray(inp_target, dtype=np.uint8)
            inp_pred = np.ascontiguousarray(inp_pred, dtype=np.uint8)

            for land in targ:
                x,y,z = land
                x = int(x)
                y = int(y)
                cv2.circle(inp_target, (x,y), 1, (0,255,0), -1)

            for land in det:
                x,y,z = land
                x = int(x)
                y = int(y)
                cv2.circle(inp_pred, (x,y), 1, (0,255,0), -1)

            final_img = np.zeros((192, 192*2, 3))
            final_img[:192, :192,:] = inp_target
            final_img[:192,192:,:] = inp_pred

            cv2.imshow('Ground vs Output', np.array(final_img, dtype = np.uint8))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            # cv2.imwrite("./InferenceResults/iter_"+str(i) + "_" + str(n) + ".jpg", final_img)
            print ("Done!")

            # exit()



main()