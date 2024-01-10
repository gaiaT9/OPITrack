"""
Author: Yan Gao
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms
from config import *

n_sigma=2
args = dict(

    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='./mots_finetune_car2',
    resume_path='./pointTrack_weights/processed_cityscapes_pretrained.pthCar',
    # resume_path='./mots_finetune_car2/checkpoint.pth',

    train_dataset = {
        'name': 'mots_cars',
        'kwargs': {
            # 'root_dir': kittiRoot,
            'root_dir': '/dev/shm/data', 
            'type': 'crop',
            'size': 7000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'AdjustBrightness',
                    'opts': {}
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
                {
                    'name': 'Flip',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                    }
                },
            ]),
        },
        # 'batch_size': 4,
        # 'workers': 1,
        'batch_size': 32,
        'workers': 16,
        # 'batch_size': 64,
        # 'workers': 32
    },

    val_dataset = {
        'name': 'mots_cars',
        'kwargs': {
            # 'root_dir': kittiRoot,
            'root_dir': '/dev/shm/data', 
            'type': 'val',
            # 'size': 500,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'size': (352, 1216),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        # 'batch_size': 16,
        # 'workers': 32,
        # 'batch_size': 2,
        # 'workers': 1
        'batch_size': 4,
        'workers': 2
    },

    model={
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1],
            'input_channel': 3
        }
    },

    lr=5e-4,
    milestones=[200],
    n_epochs=200,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSeg2Loss',

    # instance synthesis args
    min_pixel=160,
    threshold=0.94,
    with_uv=True,
    n_sigma=2
)


def get_args():
    return copy.deepcopy(args)
