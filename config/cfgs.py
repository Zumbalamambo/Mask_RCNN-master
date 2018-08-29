# -*- coding: utf-8 -*-
"""
Created on 2018/8/7 9:31
@author: gzp
Func: current configuration file
"""
import os
import sys
sys.path.append('../')
from utils.utils import make_dir
root_path = os.path.abspath('../')

# ========================================================================
# for data set
dataset_name = "multibox_roi"  # multibox_roi, Box
dataset_root_path = '/home/gezhipeng/workspace/dataset/multibox_roi_ultimate'
rgb_train_dir = os.path.join(dataset_root_path, 'rgb_train')
rgb_val_dir = os.path.join(dataset_root_path, 'rgb_valid')
mask_dir = os.path.join(dataset_root_path, 'mask')
yaml_dir = os.path.join(dataset_root_path, 'yaml')

image_min_dim = 512
image_max_dim = 512
# ========================================================================
# for training or testing
basenet_name = "Resnet101"
train_gpus = "0, 1"
images_per_gpu = 1
test_gpus = "2"
train_from_coco = True

training_version = "{}_{}_V1".format(dataset_name, basenet_name)
log_dir = os.path.join(root_path, 'logs', training_version)
output_dir = os.path.join(root_path, 'output', training_version)
val_result_dir = os.path.join(output_dir, 'valid_result')
val_result_pr_dir = os.path.join(output_dir, 'PR_result')

def init():
    make_dir(output_dir)
    make_dir(log_dir)
    make_dir(val_result_dir)
    make_dir(val_result_pr_dir)
if __name__ == '__main__':
    init()