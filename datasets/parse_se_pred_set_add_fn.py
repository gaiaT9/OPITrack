# script for seq sample dataset generate

import pickle
import numpy as np
import os
import pycocotools.mask as rletools
import mxnet as mx
from mxnet import nd
import cv2
from PIL import Image
import multiprocessing
import torch
from file_utils import *
from config import *


SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]


def getImgLabel(img_path, label_path):
    img = cv2.imread(img_path)
    label = np.array(Image.open(label_path))
    # for car
    mask = np.logical_and(label >= label_id * 1000, label < (label_id + 1) * 1000)
    # for person
    # mask = np.logical_and(label >= 2 * 1000, label < (2 + 1) * 1000)
    label[(1-mask).astype(np.bool)] = 0
    return img, label

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

def matching_pred(pred, gt):
    pred_unique = np.unique(pred)
    gt_unique = np.unique(gt)
    pred = nd.array(pred)
    gt = nd.array(gt)
    n_pred = pred_unique.shape[0] - 1
    # if no prediction ignore
    if n_pred == 0:
        return None

    n_gt = gt_unique.shape[0] - 1
    iou_matrix = nd.zeros((n_pred, n_gt))

    if n_gt == 0:
        return (pred * -1).asnumpy()
    
    for i in range(pred_unique.shape[0] - 1):
        for j in range(gt_unique.shape[0] - 1):
            p_id = pred_unique[i + 1]
            g_id = gt_unique[j + 1]
            cur_pred = pred == p_id
            cur_gt = gt == g_id
            iou_matrix[i, j] = (cur_pred * cur_gt).sum() / (((cur_pred + cur_gt) > 0).sum() + np.spacing(1))
    pred_match, gt_match = nd.contrib.bipartite_matching(iou_matrix, threshold=0.5)
    pred_match = pred_match.asnumpy().astype(np.int)
    gt_match = gt_match.asnumpy().astype(np.int)
    # process matched pred (TP & FP)
    for i in range(pred_unique.shape[0] - 1):
        raw_pid = pred_unique[i + 1]
        if pred_match[i] == -1:
            pred = nd.where(pred == raw_pid, pred * -1, pred)
        else:
            new_pid = gt_unique[pred_match[i] + 1]
            pred = nd.where(pred == raw_pid, nd.ones_like(pred) * new_pid, pred)
    # process unmatched gt (FN)
    for i in range(gt_match.shape[0]):
        if gt_match[i] == -1:
            gt_id = gt_unique[i + 1]
            pred = nd.where(gt == gt_id, nd.ones_like(pred) * gt_id, pred)
    return pred.asnumpy().astype(np.int)

def maskPool(info):
    name = info['name']
    img, label = getImgLabel(info['img'], info['label'])
    # find prediction
    if os.path.isfile(os.path.join(se_prediction_dir[0], info['pred'])):
        pred_path = os.path.join(se_prediction_dir[0], info['pred'])
    elif os.path.isfile(os.path.join(se_prediction_dir[1], info['pred'])):
        pred_path = os.path.join(se_prediction_dir[1], info['pred'])

    if np.unique(label).shape[0] == 1:
        return
    pred = load_pickle(pred_path)
    # if matched, pred_label = gt_label
    # if unmatched, pred_label = - pred_label
    mapped_pred = matching_pred(pred, label)
    # if no prediction skip
    if mapped_pred is None:
        return 

    instList = []
    obj_ids = np.unique(mapped_pred).tolist()
    obj_ids.remove(0)
    for id in obj_ids:
        mask = (mapped_pred == id).astype(np.uint8)
        maskX = (mapped_pred > 0).astype(np.uint8) * 2
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
    save_pickle2(os.path.join(target_dir, name), instList)

def getPairs(id):
    label_root = kittiRoot + "/instances/" + id
    image_root = label_root.replace('/instances', '/images')

    image_list = make_dataset(image_root, suffix='.png')
    image_list.sort()
    label_list = make_dataset(label_root, suffix='.png')
    label_list.sort()

    infos = []
    for ind, image_path in enumerate(image_list):
        infos.append({'name': id + '_' + str(ind) + '.pkl', 'img': image_list[ind], 'label': label_list[ind], 
                      'pred': id + '_' + str(ind) + '.pkl'})

    # decode all frames
    if len(infos) > 0:
        # debug code
        [maskPool(i) for i in infos]
        # pool = multiprocessing.Pool(processes=32)
        # pool.map(maskPool, infos)
        # pool.close()


if __name__ == '__main__':
    ex = 0.2
    expand_ratio = [ex, ex]

    # for car
    # label_id = 1  # car
    # target_dir = 'dataImgPredTrackEnvDB'
    # se_prediction_dir = ['car_SE_train_prediction', 'car_SE_val_prediction']

    # for person
    label_id = 2  # person
    target_dir = 'dataPersonsPredTrackEnvDB'
    se_prediction_dir = ['person_SE_train_prediction', 'person_SE_val_prediction']

    remove_and_mkdir(target_dir)
    for i in SEQ_IDS_TRAIN:
        getPairs(i)
    for i in SEQ_IDS_VAL:
        getPairs(i)

    