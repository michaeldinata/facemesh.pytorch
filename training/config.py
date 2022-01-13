'''
Add Model Configurations here
'''

# model configuration
gpu_id = 0
batch_size = 16
display = 100
apply_cropping = True
crop_size = 192
train_batch_size = 16
eval_batch_size = 16
test_batch_size = 16
num_workers = 1
run_name = 'facemesh'

# training configuration
optimizer_type = 'SGD'
momentum = 0.9
weight_decay = 5e-4
use_nesterov = True
start_epoch = 0
n_epochs = 10
lr_type = 'step'
gamma = 0.9
stepsize = 2
init_lr = 0.01

# directories
train_path_prefix = '../dataset/training_data'
eval_path_prefix = '../dataset/eval_data'
test_path_prefix = '../dataset/test_data'
write_res_prefix = '../data/res/'
flip_reflect = '../data/reflect.txt'
resume_model_path = '../facemesh.pth'