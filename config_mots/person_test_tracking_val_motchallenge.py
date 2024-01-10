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

args = dict(

    cuda=True,
    display=False,

    save=True,
    save_dir='./tracks_person_pointtrack_val/',
    checkpoint_path='./person_finetune_tracking/checkpoint.pth',
    # checkpoint_path='./pointTrack_weights/PointTrack.pthCar',
    run_eval=True,

    car=False,

    dataset={
        'name': 'mots_challenge_track_person_val',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
            'num_points': 1500,
            'box': True,
            'gt': False,
            'category': True,
            'ex': 0.2
        },
        'batch_size': 1,
        'workers': 32
    },

    model={
        'name': 'tracker_offset_emb',
        'kwargs': {
            'num_points': 1000,
            'margin': 0.2,
            'border_ic': 3,
            'env_points': 500,
            'outputD': 32,
            'category': True
        }
    },
    max_disparity=192.0,
    with_uv=True
)


def get_args():
    return copy.deepcopy(args)
