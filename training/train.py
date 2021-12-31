import argparse
import os
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


def GetDatasets(config):
    apply_cropping = config.apply_cropping
    dsets = {}

    dsets['train'] = ImageList(path=config.train_path_prefix,
                            transform=prep.image_train(crop_size=config.crop_size),
                            target_transform=prep.land_transform(img_size=config.crop_size,flip_reflect=np.loadtxt(config.flip_reflect)))

    dsets['test'] = ImageList(path=config.test_path_prefix, 
                            phase='test',
                            transform=prep.image_test(crop_size=config.crop_size),
                            target_transform=prep.land_transform(img_size=config.crop_size,flip_reflect=np.loadtxt(config.flip_reflect)))
    return dsets

def GetDataLoaders(config, dsets):
    dset_loaders = {}
    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)


    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)
    return dset_loaders

# Initialize Model Here
def InitializeModel(config):
    model = network.FaceMesh()
    return model

def ResumeModel(config, model):
    model.load_state_dict(torch.load(''))

def InitializeOptimizer(config, model):
    optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}
    model_parameter_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1}]
    optimizer = optim_dict[config.optimizer_type](itertools.chain(model_parameter_list),lr=1.0, momentum=config.momentum, weight_decay=config.weight_decay,nesterov=config.use_nesterov)
    return optimizer

def TakeSnapshot(config, model, epoch):
    print('taking snapshot ...')
    torch.save(model.state_dict(),'')


def main(config):
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
    
    if config.start_epoch > 0:
        print('resuming model from epoch %d' %(config.start_epoch))
        ResumeModel(config, model)
    print ("Network fetch successful!")

    if use_gpu:
        model = model.cuda()

    print ("Initializing Optimizer!!")    
    optimizer = InitializeOptimizer(config, model)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    lr_scheduler = lr_schedule.schedule_dict[config.lr_type]


    '''
        Training Loop Starts Here!!!!
    '''
    count = 0
    for epoch in range(config.start_epoch, config.n_epochs + 1):
        if epoch > config.start_epoch:
            TakeSnapshot(config, model, epoch)

        if epoch > config.start_epoch:
            print('testing ...')
            model.train(False)
            f1score_arr, acc_arr, mean_error, failure_rate = detection_eval(dset_loaders['test'], model, use_gpu=use_gpu)
            print('epoch =%d, f1 score mean=%f, accuracy mean=%f, mean error=%f, failure rate=%f'
                  % (epoch, f1score_arr.mean(), acc_arr.mean(), mean_error, failure_rate))
            model.train(True)

        if epoch >= config.n_epochs:
            break

        for i, batch in enumerate(dset_loaders['train']):
            if (i % config.display == 0 and count > 0):
                print('[epoch = %d][iter = %d][total_loss = %f]' % (epoch, i,total_loss.data.cpu().numpy()))
                print('learning rate = %f' % (optimizer.param_groups[0]['lr']))
                print('the number of training iterations is %d' % (count))

            input, target = batch

            if use_gpu:
                input = input.cuda()
                target = target.long().cuda()
            else:
                au = au.long()

            optimizer = lr_scheduler(param_lr, optimizer, epoch, config.gamma, config.stepsize, config.init_lr)
            optimizer.zero_grad()

            pred = model(input)
            ## Define Loss Function Here

            ####
            total_loss.backward()
            optimizer.step()
            count = count + 1

    res_file.close()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    main(config)