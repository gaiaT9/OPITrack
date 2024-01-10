"""
Author: Yan Gao
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

import cv2
import numpy as np
import PIL.Image as Image
import multiprocessing
import sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from file_utils import *
from config import *

from pycocotools import mask as maskUtils


SEQ_IDS_TRAIN = ["MOTS20-%02d" % idx for idx in [2, 5, 9]]
SEQ_IDS_VAL = ["MOTS20-%02d" % idx for idx in [11]]


def getImgLabel(img_path, label):
    img = cv2.imread(img_path)
    
    full_mask = None
    for l in label:
        _, track_id, _, h, w, maskstr = l.split(' ')
        mask = maskUtils.decode({'size': [int(h), int(w)], 'counts': maskstr})
        track_id = int(track_id)
        if full_mask is None:
            full_mask = np.zeros((int(h), int(w)), dtype=np.int)
        # no overlap between GT
        full_mask += mask * track_id
    return img, full_mask


def get_grids(mask):
    # grid need to be in [-1,1], and uv rather than vu
    h, w = mask.shape
    s = torch.zeros(h)
    t = torch.zeros(h)
    for i in range(h):
        us = torch.nonzero(mask[i])
        if us.shape[0] > 0:
            t[i] = us.min()
            s[i] = (us.max() - us.min() + 1) / float(w)
        else:
            t[i] = 0
            s[i] = 1.0
    xTemplate = torch.arange(w).unsqueeze(0).repeat(h, 1)
    us = xTemplate * s.unsqueeze(-1) + t.unsqueeze(-1)
    vs = torch.arange(h).float().unsqueeze(-1).repeat(1, w)
    grid = torch.cat([us.unsqueeze(-1) / (w-1), vs.unsqueeze(-1) / (h-1)], dim=-1)
    return (grid - 0.5) * 2


def maskPool(info):
    name = info['name']
    img, label = getImgLabel(info['img'], info['label'])
    if np.unique(label).shape[0] == 1:
        return

    instList = []
    obj_ids = np.unique(label).tolist()[1:]
    for id in obj_ids:
        mask = (label == id).astype(np.uint8)
        maskX = (label > 0).astype(np.uint8) * 2
        maskX[mask > 0] = 1
        h, w = mask.shape
        vs, us = np.nonzero(mask)
        v0, v1 = vs.min(), vs.max()
        vlen = v1 - v0
        u0, u1 = us.min(), us.max()
        ulen = u1 - u0
        # enlarge box by expand_ratio
        v0 = max(0, v0 - int(expand_ratio[0] * vlen))
        v1 = min(v1 + int(expand_ratio[0]* vlen), h - 1)
        u0 = max(0, u0 - int(expand_ratio[1] * ulen))
        u1 = min(u1 + int(expand_ratio[1] * ulen), w - 1)
        inst_id = id
        sp = [v0, u0]
        imgCrop = img[v0:v1 + 1, u0:u1 + 1]
        maskCrop = mask[v0:v1 + 1, u0:u1 + 1]
        maskX = maskX[v0:v1 + 1, u0:u1 + 1]

        instList.append({'inst_id': inst_id, 'sp': sp, 'img': imgCrop, 'mask': maskCrop, 'maskX': maskX})
    save_pickle2(os.path.join(outF, name), instList)


def getPairs(id):
    motchallege_root = '/data/dataset/motchallenge/MOTS/train'
    label_root = os.path.join(motchallege_root, id, 'gt', 'gt.txt')
    image_root = os.path.join(motchallege_root, id, 'img1')

    image_list = make_dataset(image_root, suffix='.jpg')
    image_list.sort()
    label_list = open(label_root, 'r').read().split('\n')
    label_list.remove('')
    labels = {}
    for l in label_list:
        fid = int(l.split(' ')[0])
        if not fid in labels:
            labels[fid] = []
        labels[fid].append(l)

    infos = []
    for ind, image_path in enumerate(image_list):
        # time id start from 1 in motchallenge dataset
        infos.append({'name': id + '_' + str(ind) + '.pkl', 'img': image_list[ind], 'label': labels[ind + 1]})

    # decode all frames
    if len(infos) > 0:
        pool = multiprocessing.Pool(processes=32)
        pool.map(maskPool, infos)
        pool.close()


ex = 0.2
expand_ratio = [ex, ex]

label_id = 2  # pedestrian
outF = kittiRoot + 'MOTChallengeTrackDB/'
remove_and_mkdir(outF)
for i in SEQ_IDS_TRAIN:
    getPairs(i)
for i in SEQ_IDS_VAL:
    getPairs(i)
