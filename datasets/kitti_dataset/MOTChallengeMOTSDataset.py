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

SEQ_IDS_TRAIN = ["MOTS20-%02d" % idx for idx in [2, 5, 9]]
SEQ_IDS_VAL = ["MOTS20-%02d" % idx for idx in [11]]
TIMESTEPS_PER_SEQ = {"MOTS20-02": 600, "MOTS20-05": 837, "MOTS20-09": 525, "MOTS20-11": 900}
SEQ_IDS_TEST =  ["MOTS20-%02d" % idx for idx in [1, 6, 7, 12]]
TIMESTEPS_PER_SEQ_TEST = {"MOTS20-01": 450, "MOTS20-06": 1194, "MOTS20-07": 500, "MOTS20-12": 900}

MOTSROOT = '/data/dataset/motchallenge/MOTS'

class MOTSTrackPersonValOffset(Dataset):
    def __init__(self, root_dir='./', type="train", num_points=250, transform=None, random_select=False, az=False,
                 border=False, env=False, gt=True, box=False, test=False, category=False, ex=0.2):

        print('MOTS Dataset created')
        # type = 'training' if type in 'training' else 'testing'
        self.type = type
        # assert self.type == 'testing'

        self.transform = transform
        if type in 'validate':
            ids = SEQ_IDS_VAL
            timestamps = TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(MOTSROOT, 'train')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/motchallenge_pred')
        elif type in 'training':
            ids = SEQ_IDS_TRAIN
            timestamps = TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(MOTSROOT, 'train')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/motchallenge_pred')
        elif type in 'testing':
            ids = SEQ_IDS_TEST
            timestamps = TIMESTEPS_PER_SEQ_TEST
            self.image_root = os.path.join(MOTSROOT, 'test')
            self.mots_root = os.path.join(systemRoot, 'OPITrack/motchallenge_pred_test')
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
        imgPath = os.path.join(self.image_root, subf, 'img1', '%06d.jpg' % int(float(frameCount) + 1))
        img = cv2.imread(imgPath)

        sample = {}
        # sample['name'] = imgPath
        sample['name'] = os.path.join(self.image_root, subf, '%06d.jpg' % int(float(frameCount) + 1))
        sample['masks'] = []
        sample['points'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        inds = np.unique(instance_map).tolist()[1:]
        label = (instance_map > 0).astype(np.uint8) * 2
        for inst_id in inds:
            mask = (instance_map == inst_id)
            sample['xyxys'].append(self.get_xyxy_from_mask(mask))
            sample['masks'].append(np.array(mask)[np.newaxis])
            mask, img_, maskX, sp = self.get_crop_from_mask(mask, img, label.copy())
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
            choices = np.random.choice(pointUVs.shape[0], fg_num)
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

class MOTSTrackPersonTrain(Dataset):
    def __init__(self, root_dir='./', type="train", num_points=250, transform=None, random_select=False, batch_num=8,
                 shift=False, size=3000, sample_num=30, nearby=1, trainval=False, category=False):
        print('MOTS Dataset created')
        type = 'training' if type in 'training' else 'testing'
        if trainval:
            self.squence = SEQ_IDS_TRAIN + SEQ_IDS_VAL
            print("train with training and val set")
        else:
            self.squence = SEQ_IDS_TRAIN if type == 'training' else SEQ_IDS_VAL
        self.type = type

        self.transform = transform
        db_dir = kittiRoot + 'MOTChallengeTrackDB/'
        self.dbDict = {}
        for id in self.squence:
            image_root = MOTSROOT + "/train/" + id + '/img1'
            image_list = make_dataset(image_root, suffix='.jpg')
            image_list.sort()
            infos = {}
            for ind, image_path in enumerate(image_list):
                pkl_path = os.path.join(db_dir, id + '_' + str(ind) + '.pkl')
                if os.path.isfile(pkl_path):
                    infos[ind] = load_pickle(pkl_path)
            self.dbDict[id] = infos

        self.mots_car_instances = self.getInstanceFromDB(self.dbDict)
        print('dbDict Loaded, %s instances' % len(self.mots_car_instances))
        self.batch_num = batch_num

        self.inst_names = list(self.mots_car_instances.keys())
        self.inst_num = len(self.inst_names)
        self.real_size = size
        self.mots_class_id = 1
        self.vMax, self.uMax = 375.0, 1242.0
        self.offsetMax = 128.0
        self.num_points = num_points
        self.random = random_select
        self.shift = shift
        self.frequency = 1
        self.sample_num = sample_num
        self.nearby = nearby
        self.category = category
        self.category_embedding = np.array(category_embedding, dtype=np.float32)

    def getInstanceFromDB(self, dbDict):
        allInstances = {}
        for k, fs in dbDict.items():
            # current video k
            # num_frames = self.TIMESTEPS_PER_SEQ[k]
            if not k in self.squence:
                continue
            for fi, f in fs.items():
                frameCount = fi
                for inst in f:
                    inst_id = k + '_' + str(inst['inst_id'])
                    newDict = {'frame': frameCount, 'sp': inst['sp'], 'img': inst['img'], 'mask': inst['mask'], 'maskX': inst['maskX']}
                    if not inst_id in allInstances.keys():
                        allInstances[inst_id] = [newDict]
                    else:
                        allInstances[inst_id].append(newDict)
        return allInstances

    def __len__(self):
        return self.real_size

    def get_data_from_mots(self, index):
        # sample ? instances from self.inst_names
        inst_names_inds = random.sample(range(len(self.inst_names)), self.sample_num)
        inst_names = [self.inst_names[el] for el in inst_names_inds]
        pickles = [self.mots_car_instances[el] for el in inst_names]

        sample = {}
        sample['mot_im_name0'] = index
        sample['points'] = []
        sample['labels'] = []
        sample['imgs'] = []
        sample['inds'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        for pind, pi in enumerate(pickles):
            # instance sample code
            inst_id = pind + 1
            inst_length = len(pi)
            if inst_length > 2:
                mid = random.choice(range(1, inst_length - 1))
                nearby = random.choice(range(1, self.nearby+1))
                start, end = max(0, mid - nearby), min(inst_length - 1, mid + nearby)
                pis = [pi[start], pi[mid], pi[end]]
            else:
                start, end = 0, 1
                pis = pi[start:end + 1]
            # instance processing code
            for ii, inst in enumerate(pis):
                img = inst['img']
                mask = inst['mask'].astype(np.bool)
                maskX = inst['maskX']
                sp = inst['sp']
                assert (~mask).sum() > 0

                ratio = 2.0
                bg_num = int(self.num_points / (ratio + 1))
                fg_num = self.num_points - bg_num

                # get center
                vs_, us_ = np.nonzero(mask)
                vc, uc = vs_.mean(), us_.mean()

                vs, us = np.nonzero(~mask)
                vs = (vs - vc) / self.offsetMax
                us = (us - uc) / self.offsetMax
                rgbs = img[~mask] / 255.0
                if self.shift:
                    # us += (random.random() - 0.5) * 0.05  # -0.025~0.025
                    vs += np.random.normal(0, 0.001, size=vs.shape)  # random jitter
                    us += np.random.normal(0, 0.001, size=us.shape)  # random jitter
                cats = maskX[~mask]
                cat_embds = self.category_embedding[cats]
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis], cat_embds], axis=1)
                choices = np.random.choice(pointUVs.shape[0], bg_num)
                points_bg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)

                vs = (vs_ + sp[0]) / self.vMax
                us = (us_ + sp[1]) / self.uMax  # to compute the bbox position
                sample['xyxys'].append([us.min(), vs.min(), us.max(), vs.max()])

                vs = (vs_ - vc) / self.offsetMax
                us = (us_ - uc) / self.offsetMax
                rgbs = img[mask.astype(np.bool)] / 255.0
                if self.shift:
                    # us += (random.random() - 0.5) * 0.05  # -0.025~0.025
                    vs += np.random.normal(0, 0.001, size=vs.shape)  # random jitter
                    us += np.random.normal(0, 0.001, size=us.shape)  # random jitter
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
                choices = np.random.choice(pointUVs.shape[0], fg_num)
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
                points_fg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
                points_fg = np.concatenate(
                    [points_fg, np.zeros((points_fg.shape[0], points_fg.shape[1], 3), dtype=np.float32)], axis=-1)

                sample['points'].append(np.concatenate([points_fg, points_bg], axis=1))
                sample['labels'].append(np.array(inst_id)[np.newaxis])
                sample['envs'].append(fg_num)

        sample['points'] = np.concatenate(sample['points'], axis=0)
        sample['envs'] = np.array(sample["envs"], dtype=np.int32)
        sample['labels'] = np.concatenate(sample['labels'], axis=0)
        sample['xyxys'] = np.array(sample["xyxys"], dtype=np.float32)
        return sample

    def __getitem__(self, index):
        # select nearby images from mots
        while 1:
            try:
                sample = self.get_data_from_mots(index)
                break
            except:
                pass
        # sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


