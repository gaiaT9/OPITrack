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
from tqdm import tqdm
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from file_utils import *
from config import *

from AMB.calc_kalman_baseline import warp_mask_with_flow


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




def maskPool(info):
    name = info['name']
    img, label = getImgLabel(info['img'], info['label'])
    seq, tid = info['img'].split('/')[-2:]
    tid = int(tid.split('.')[0])
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
    # get prev frame image and calc warping
    prev_img_path = '/'.join(info['img'].split('/')[:-2] + [seq, '%06d.png' % (tid - 1)])
    prev_label_path = '/'.join(info['label'].split('/')[:-2] + [seq, '%06d.png' % (tid - 1)])
    
    if os.path.exists(prev_img_path):
        _, prev_label = getImgLabel(prev_img_path, prev_label_path)
        flow = pickle.load(open('kitti_flow/%s_%d.pkl' % (seq, tid - 1), 'rb'))
        # only calc warping of current exist target
        if np.unique(prev_label).shape[0] != 1:
            # perform warp to every instance from prev frame
            global_warp_mask = np.zeros_like(prev_label)
            for id in np.unique(prev_label)[1:]:
                bin_mask = prev_label == id
                warp_bin_mask = warp_mask_with_flow(bin_mask, flow)
                global_warp_mask[warp_bin_mask] = id
            for id in obj_ids:
                prev_mask = (global_warp_mask == id).astype(np.uint8)
                if prev_mask.sum() == 0:
                    continue
                mask = prev_mask
                maskX = (global_warp_mask > 0).astype(np.uint8) * 2
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

                # delete the same id object
                for obj_idx in range(len(instList)):
                    if instList[obj_idx] == inst_id:
                        del instList[obj_idx]
                        break
                instList.append({'inst_id': inst_id, 'sp': sp, 'img': imgCrop, 'mask': maskCrop, 'maskX': maskX})


    save_pickle2(os.path.join(outF, name), instList)


def getPairs(id):
    label_root = kittiRoot + "/instances/" + id
    image_root = label_root.replace('/instances', '/images')

    image_list = make_dataset(image_root, suffix='.png')
    image_list.sort()
    label_list = make_dataset(label_root, suffix='.png')
    label_list.sort()

    infos = []
    for ind, image_path in enumerate(image_list):
        infos.append({'name': id + '_' + str(ind) + '.pkl', 'img': image_list[ind], 'label': label_list[ind]})

    # decode all frames
    if len(infos) > 0:
        pool = multiprocessing.Pool(processes=32)
        pool.map(maskPool, infos)
        pool.close()
        # [maskPool(f) for f in infos]


ex = 0.2
expand_ratio = [ex, ex]

label_id = 1  # car
outF = kittiRoot + 'ImgTrackEnvDB/'
remove_and_mkdir(outF)
for i in tqdm(SEQ_IDS_TRAIN):
    getPairs(i)
for i in tqdm(SEQ_IDS_VAL):
    getPairs(i)

# label_id = 2  # pedestrian
# outF = kittiRoot + 'PersonsTrackDB/'
# remove_and_mkdir(outF)
# for i in SEQ_IDS_TRAIN:
#     getPairs(i)
# for i in SEQ_IDS_VAL:
#     getPairs(i)
