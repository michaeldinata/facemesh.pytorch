import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
import itertools

import config
import network
import pre_process as prep
import lr_schedule
from util import *
from data_list import ImageList

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

# Initialize Model Here
def InitializeModel(config):
    model = network.FaceMesh()
    return model

def ResumeModel(config, model):
    model.load_state_dict(torch.load(config.resume_model_path))

def InitializeOptimizer(config, model):
    optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}
    model_parameter_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1.0}]
    optimizer = optim_dict[config.optimizer_type](itertools.chain(model_parameter_list),lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay,nesterov=config.use_nesterov)
    return optimizer

def TakeSnapshot(config, model, epoch):
    print('taking snapshot ...')
    torch.save(model.state_dict(),'./Snapshots/' + str(epoch) + '.pth')


def main(config):
    MakeDir("./Snapshots/")

    ## set loss criterion
    use_gpu = torch.cuda.is_available()
    
    ## prepare data
    print ("Preparing Data!")
    dsets = GetDatasets(config)
    dset_loaders = GetDataLoaders(config, dsets)
    print ("Done! Preparing Data")
    
    ## set network modules
    print ("Getting Networks!!")
    model = InitializeModel(config)
    # model.load_state_dict(torch.load('facemesh.pth'))


    if config.start_epoch > 0:
        print('resuming model from epoch %d' %(config.start_epoch))
        ResumeModel(config, model)
    print ("Network fetch successful!")

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if use_gpu:
        model = model.cuda()

    print ("Initializing Optimizer!!")    
    optimizer = InitializeOptimizer(config, model)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    lr_scheduler = lr_schedule.schedule_dict[config.lr_type]

    # res_file = open(
    #     config.write_res_prefix + config.run_name + '/pred_' + str(config.start_epoch) + '.txt', 'w')

    '''
        Training Loop Starts Here!!!!
    '''
    count = 0
    for epoch in range(config.start_epoch, config.n_epochs + 1):
        if epoch > config.start_epoch and epoch % 5 == 0:
            TakeSnapshot(config, model, epoch)

        if epoch > config.start_epoch:
            print('testing ...')
            model.train(False)
            mse_loss = detection_eval(dset_loaders['eval'], model, use_gpu=use_gpu)
            print('mean_error=%f' % (mse_loss))
            model.train(True)

        if epoch >= config.n_epochs:
            break
        
        total_mse_batch = 0
        epoch_iterations = 0
        for i, batch in enumerate(dset_loaders['train']):
            # print(optimizer.param_groups[0]['lr'])
            if (i % config.display == 0 and count > 0):
                print('[epoch = %d][iter = %d][total_loss = %f]' % (epoch, i,total_loss.data.cpu().numpy()))
                print('learning rate = %f' % (optimizer.param_groups[0]['lr']))
                print('the number of training iterations is %d' % (count))

            input, target = batch

            if use_gpu:
                input = input.float().cuda()
                target = target.float().cuda()
            else:
                target = target.float()

            optimizer = lr_scheduler(param_lr, optimizer, epoch, config.gamma, config.stepsize, config.init_lr)

            optimizer.zero_grad()
            # print (input.size())
            pred = model(input)
            ## Define Loss Function Here
            detections, confidences = pred
            # print (detections)
            # print (target)
            # print (total_loss)
            # exit()
            total_loss = F.mse_loss(detections, target)
            # print (detections)
            # print (target)
            # print (total_loss)
            total_mse_batch += total_loss
            epoch_iterations += 1
            total_loss.backward()
            optimizer.step()
            count = count + 1
        print ("-------------------------------")
        print ("Total Loss Epoch : " + str(total_mse_batch/epoch_iterations))
        print("--------------------------------")
    # res_file.close()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    main(config)