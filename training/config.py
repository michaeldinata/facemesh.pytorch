'''
Add Model Configurations here
'''

gpu_id = 0
batch_size = 16
display = 100
apply_cropping = True
crop_size = 256

flip_reflect = '../data/reflect.txt'

train_path_prefix = '../training_data'
train_batch_size = 16

test_path_prefix = '../training_data'
eval_batch_size = 16

num_workers = 1

optimizer_type = 'SGD'
momentum = 0.9
weight_decay = 5e-4
use_nesterov = True
start_epoch = 0
n_epochs = 50
lr_type = 'lambda'
gamma = 0.7
stepsize = 2
init_lr = 0.001