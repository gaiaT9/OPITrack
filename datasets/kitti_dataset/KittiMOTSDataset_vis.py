"""
Author: Yan Gao
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random
import numpy as np
import math
from PIL import Image
from skimage.segmentation import relabel_sequential
import torch
from torch.utils.data import Dataset
import cv2
from config import *
from file_utils import *
from utils.mots_util import *
import torch.nn.functional as F


class MOTSTrackCarsValOffset(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}

    def __init__(self, root_dir='./', type="train", num_points=250, transform=None, random_select=False, az=False,
                 border=False, env=False, gt=True, box=False, test=False, category=False, ex=0.2):

        print('MOTS Dataset created')
        # type = 'training' if type in 'training' else 'testing'
        self.type = type
        # assert self.type == 'testing'

        self.transform = transform
        if type in 'validate':
            ids = self.SEQ_IDS_VAL
            timestamps = self.TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(kittiRoot, 'images')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/car_SE_val_prediction')
        elif type in 'training':
            ids = self.SEQ_IDS_TRAIN
            timestamps = self.TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(kittiRoot, 'images')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/car_SE_train_prediction')
        elif type in 'testing':
            ids = self.SEQ_IDS_TEST
            timestamps = self.TIMESTEPS_PER_SEQ_TEST
            self.image_root = os.path.join(kittiRoot, 'testing/image_02/')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/car_SE_test_prediction')
        else:
            raise ValueError('Unknown type: %s' % type)

        print('use ', self.mots_root)
        self.batch_num = 2

        self.mots_car_sequence = []
        for valF in ids:
            # frame number
            nums = timestamps[valF]
            for i in range(nums):
                pklPath = os.path.join(self.mots_root, valF + '_' + str(i) + '.pkl')
                if os.path.isfile(pklPath):
                    self.mots_car_sequence.append(pklPath)

        self.real_size = len(self.mots_car_sequence)
        self.mots_num = len(self.mots_car_sequence)
        self.mots_class_id = 1
        self.expand_ratio = ex
        self.vMax, self.uMax = 375.0, 1242.0
        self.num_points = num_points
        self.env_points = 200
        self.random = random_select
        self.az = az
        self.border = border
        self.env = env
        self.box = box
        self.offsetMax = 128.0
        self.category = category
        self.category_embedding = np.array(category_embedding, dtype=np.float32)
        print(self.mots_root)

    def __len__(self):
        return self.real_size

    def get_crop_from_mask(self, mask, img, label):
        label[mask] = 1
        vs, us = np.nonzero(mask)
        h, w = mask.shape
        v0, v1 = vs.min(), vs.max() + 1
        vlen = v1 - v0
        u0, u1 = us.min(), us.max() + 1
        ulen = u1 - u0
        # enlarge box by 0.2
        v0 = max(0, v0 - int(self.expand_ratio * vlen))
        v1 = min(v1 + int(self.expand_ratio * vlen), h - 1)
        u0 = max(0, u0 - int(self.expand_ratio * ulen))
        u1 = min(u1 + int(self.expand_ratio * ulen), w - 1)
        return mask[v0:v1, u0:u1], img[v0:v1, u0:u1], label[v0:v1, u0:u1], (v0, u0)

    def get_xyxy_from_mask(self, mask):
        vs, us = np.nonzero(mask)
        y0, y1 = vs.min(), vs.max()
        x0, x1 = us.min(), us.max()
        return [x0/self.uMax, y0/self.vMax, x1/self.uMax, y1/self.vMax]

    def get_data_from_mots(self, index):
        # random select and image and the next one
        path = self.mots_car_sequence[index]
        instance_map = load_pickle(path)
        subf, frameCount = os.path.basename(path)[:-4].split('_')
        imgPath = os.path.join(self.image_root, subf, '%06d.png' % int(float(frameCount)))
        img = cv2.imread(imgPath)

        sample = {}
        sample['name'] = imgPath
        sample['masks'] = []
        sample['points'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        sample['img_patches'] = []
        sample['sample_pos_fg'] = []
        sample['sample_pos_bg'] = []
        inds = np.unique(instance_map).tolist()[1:]
        label = (instance_map > 0).astype(np.uint8) * 2
        for inst_id in inds:
            mask = (instance_map == inst_id)
            sample['xyxys'].append(self.get_xyxy_from_mask(mask))
            sample['masks'].append(np.array(mask)[np.newaxis])
            mask, img_, maskX, sp = self.get_crop_from_mask(mask, img, label.copy())
            sample['img_patches'].append(img_)
            # fg/bg ratio
            ratio = 2.0
            # ratio = max(mask.sum() / (~mask).sum(), 2.0)
            bg_num = int(self.num_points / (ratio + 1))
            fg_num = self.num_points - bg_num

            vs_, us_ = np.nonzero(mask)
            vc, uc = vs_.mean(), us_.mean()

            vs = (vs_ - vc) / self.offsetMax
            us = (us_ - uc) / self.offsetMax
            rgbs = img_[mask] / 255.0
            pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
            vu_ = np.concatenate([vs_[:, np.newaxis], us_[:, np.newaxis]], axis=1)
            choices = np.random.choice(pointUVs.shape[0], fg_num)
            chosen_vu_ = vu_[choices]
            sample['sample_pos_fg'].append(chosen_vu_)
            points_fg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
            points_fg = np.concatenate(
                [points_fg, np.zeros((points_fg.shape[0], points_fg.shape[1], 3), dtype=np.float32)], axis=-1)

            if (~mask).sum() == 0:
                points_bg = np.zeros((1, bg_num, 8), dtype=np.float32)
            else:
                vs_, us_ = np.nonzero(~mask)
                vs = (vs_ - vc) / self.offsetMax
                us = (us_ - uc) / self.offsetMax
                rgbs = img_[~mask] / 255.0
                cats = maskX[~mask]
                cat_embds = self.category_embedding[cats]
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis], cat_embds], axis=1)
                choices = np.random.choice(pointUVs.shape[0], bg_num)
                vu_ = np.concatenate([vs_[:, np.newaxis], us_[:, np.newaxis]], axis=1)
                sample['sample_pos_bg'].append(vu_[choices])
                points_bg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
            sample['points'].append(np.concatenate([points_fg, points_bg], axis=1))
            sample['envs'].append(fg_num)

        if len(sample['points']) > 0:
            sample['points'] = np.concatenate(sample['points'], axis=0)
            sample['masks'] = np.concatenate(sample['masks'], axis=0)
            sample['envs'] = np.array(sample["envs"], dtype=np.int32)
            sample['xyxys'] = np.array(sample["xyxys"], dtype=np.float32)
        return sample

    def __getitem__(self, index):
        # select nearby images from mots
        sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample

class MOTSTrackPersonValOffset(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}

    def __init__(self, root_dir='./', type="train", num_points=250, transform=None, random_select=False, az=False,
                 border=False, env=False, gt=True, box=False, test=False, category=False, ex=0.2):

        print('MOTS Dataset created')
        # type = 'training' if type in 'training' else 'testing'
        self.type = type
        # assert self.type == 'testing'

        self.transform = transform
        if type in 'validate':
            ids = self.SEQ_IDS_VAL
            timestamps = self.TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(kittiRoot, 'images')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/person_SE_val_prediction')
        elif type in 'training':
            ids = self.SEQ_IDS_TRAIN
            timestamps = self.TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(kittiRoot, 'images')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/person_SE_train_prediction')
        elif type in 'testing':
            ids = self.SEQ_IDS_TEST
            timestamps = self.TIMESTEPS_PER_SEQ_TEST
            self.image_root = os.path.join(kittiRoot, 'testing/image_02/')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/person_SE_test_prediction')
        else:
            raise ValueError('Unknown type: %s' % type)

        print('use ', self.mots_root)
        self.batch_num = 2

        self.mots_car_sequence = []
        for valF in ids:
            # frame number
            nums = timestamps[valF]
            for i in range(nums):
                pklPath = os.path.join(self.mots_root, valF + '_' + str(i) + '.pkl')
                if os.path.isfile(pklPath):
                    self.mots_car_sequence.append(pklPath)

        self.real_size = len(self.mots_car_sequence)
        self.mots_num = len(self.mots_car_sequence)
        self.mots_class_id = 1
        self.expand_ratio = ex
        self.vMax, self.uMax = 375.0, 1242.0
        self.num_points = num_points
        self.env_points = 200
        self.random = random_select
        self.az = az
        self.border = border
        self.env = env
        self.box = box
        self.offsetMax = 128.0
        self.category = category
        self.category_embedding = np.array(category_embedding, dtype=np.float32)
        print(self.mots_root)

    def __len__(self):
        return self.real_size

    def get_crop_from_mask(self, mask, img, label):
        label[mask] = 1
        vs, us = np.nonzero(mask)
        h, w = mask.shape
        v0, v1 = vs.min(), vs.max() + 1
        vlen = v1 - v0
        u0, u1 = us.min(), us.max() + 1
        ulen = u1 - u0
        # enlarge box by 0.2
        v0 = max(0, v0 - int(self.expand_ratio * vlen))
        v1 = min(v1 + int(self.expand_ratio * vlen), h - 1)
        u0 = max(0, u0 - int(self.expand_ratio * ulen))
        u1 = min(u1 + int(self.expand_ratio * ulen), w - 1)
        return mask[v0:v1, u0:u1], img[v0:v1, u0:u1], label[v0:v1, u0:u1], (v0, u0)

    def get_xyxy_from_mask(self, mask):
        vs, us = np.nonzero(mask)
        y0, y1 = vs.min(), vs.max()
        x0, x1 = us.min(), us.max()
        return [x0/self.uMax, y0/self.vMax, x1/self.uMax, y1/self.vMax]

    def get_data_from_mots(self, index):
        # random select and image and the next one
        path = self.mots_car_sequence[index]
        instance_map = load_pickle(path)
        subf, frameCount = os.path.basename(path)[:-4].split('_')
        imgPath = os.path.join(self.image_root, subf, '%06d.png' % int(float(frameCount)))
        img = cv2.imread(imgPath)

        sample = {}
        sample['name'] = imgPath
        sample['masks'] = []
        sample['points'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        sample['img_patches'] = []
        sample['sample_pos'] = []
        inds = np.unique(instance_map).tolist()[1:]
        label = (instance_map > 0).astype(np.uint8) * 2
        for inst_id in inds:
            mask = (instance_map == inst_id)
            sample['xyxys'].append(self.get_xyxy_from_mask(mask))
            sample['masks'].append(np.array(mask)[np.newaxis])
            mask, img_, maskX, sp = self.get_crop_from_mask(mask, img, label.copy())
            sample['img_patches'].append(img_)
            # fg/bg ratio
            ratio = 2.0
            # ratio = max(mask.sum() / (~mask).sum(), 2.0)
            bg_num = int(self.num_points / (ratio + 1))
            fg_num = self.num_points - bg_num

            vs_, us_ = np.nonzero(mask)
            vc, uc = vs_.mean(), us_.mean()

            vs = (vs_ - vc) / self.offsetMax
            us = (us_ - uc) / self.offsetMax
            rgbs = img_[mask] / 255.0
            pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
            vu_ = np.concatenate([vs_[:, np.newaxis], us_[:, np.newaxis]], axis=1)
            choices = np.random.choice(pointUVs.shape[0], fg_num)
            chosen_vu_ = vu_[choices]
            sample['sample_pos'].append(chosen_vu_)
            points_fg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
            points_fg = np.concatenate(
                [points_fg, np.zeros((points_fg.shape[0], points_fg.shape[1], 3), dtype=np.float32)], axis=-1)

            if (~mask).sum() == 0:
                points_bg = np.zeros((1, bg_num, 8), dtype=np.float32)
            else:
                vs, us = np.nonzero(~mask)
                vs = (vs - vc) / self.offsetMax
                us = (us - uc) / self.offsetMax
                rgbs = img_[~mask] / 255.0
                cats = maskX[~mask]
                cat_embds = self.category_embedding[cats]
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis], cat_embds], axis=1)
                choices = np.random.choice(pointUVs.shape[0], bg_num)
                points_bg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
            sample['points'].append(np.concatenate([points_fg, points_bg], axis=1))
            sample['envs'].append(fg_num)

        if len(sample['points']) > 0:
            sample['points'] = np.concatenate(sample['points'], axis=0)
            sample['masks'] = np.concatenate(sample['masks'], axis=0)
            sample['envs'] = np.array(sample["envs"], dtype=np.int32)
            sample['xyxys'] = np.array(sample["xyxys"], dtype=np.float32)
        return sample

    def __getitem__(self, index):
        # select nearby images from mots
        sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


