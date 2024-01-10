import os, sys
import numpy as np
import pickle
from tqdm import tqdm
import time

target_dir = sys.argv[1]
# target_dir = 'dataImgPredTrackEnvDB'

# configs
# calc IoU and record >0 samples
# mark segm area drastic changed, use 1-order diff to measure
# iou_window = -1 means global iou calculations
iou_window = 2

def calc_iou(elem1, elem2):
    # y, x
    sp1 = elem1['sp']
    sp2 = elem2['sp']
    # y, x ,h, w
    box1 = sp1 + list(elem1['mask'].shape)
    box2 = sp2 + list(elem2['mask'].shape)
    top_left = [min(sp1[0], sp2[0]), min(sp1[1], sp2[1])]
    max_h = max(box1[2] + box1[0], box2[2] + box2[0]) - top_left[0]
    max_w = max(box1[3] + box1[1], box2[3] + box2[1]) - top_left[1]
    elem1_expand_mask = np.zeros((max_h, max_w))
    elem2_expand_mask = np.zeros((max_h, max_w))
    elem1_offset = [sp1[0] - top_left[0], sp1[1] - top_left[1]]
    elem2_offset = [sp2[0] - top_left[0], sp2[1] - top_left[1]]
    elem1_expand_mask[elem1_offset[0]: elem1_offset[0] + box1[2],
                      elem1_offset[1]: elem1_offset[1] + box1[3]] = elem1['mask']
    elem2_expand_mask[elem2_offset[0]: elem2_offset[0] + box2[2],
                      elem2_offset[1]: elem2_offset[1] + box2[3]] = elem2['mask']
    iou = (elem1_expand_mask * elem2_expand_mask).sum() / (((elem1_expand_mask + elem2_expand_mask) > 0).sum() + np.spacing(1))
    return iou

def warp_iou(tup):
    return calc_iou(*tup)

def process_seq(seq_data):
    tup = time.time()
    data = {}
    # load data
    for f in seq_data:
        tid = int(f.split('.')[0].split('_')[1])
        pkl = pickle.load(open(os.path.join(target_dir, f), 'rb'))
        # add iou key
        for o in pkl:
            o['iou'] = []
        data[tid] = pkl
    tids = list(data.keys())
    tids.sort()
    # iou calculations
    payload = []
    # add payload
    for i in tqdm(range(len(tids) - 1)):
        current_tid = tids[i]
        if not current_tid in tids:
            continue
        # determine range of IoU calculations
        if iou_window > 0:
            t_window = range(current_tid, current_tid + iou_window)
        else:
            t_window = range(current_tid, max(tids) + 1)
        # slide window
        for t in t_window:
            if not t in tids:
                continue
            for c_obj in data[current_tid]:
                inst_id = c_obj['inst_id']
                # skip neg id
                if inst_id < 0:
                    continue
                for t_obj in data[t]:
                    t_inst_id = t_obj['inst_id']
                    # skip neg id
                    if t_inst_id < 0:
                        continue
                    if inst_id != t_inst_id:
                        iou = payload.append((c_obj, t_obj))
    import multiprocessing as mp
    pool = mp.Pool(48)
    result = pool.map(warp_iou, payload)
    pool.close()
    # get_result
    for i in tqdm(range(len(tids) - 1)):
        current_tid = tids[i]
        if not current_tid in tids:
            continue
        # determine range of IoU calculations
        if iou_window > 0:
            t_window = range(current_tid, current_tid + iou_window)
        else:
            t_window = range(current_tid, max(tids) + 1)
        # slide window
        for t in t_window:
            if not t in tids:
                continue
            for c_obj in data[current_tid]:
                inst_id = c_obj['inst_id']
                # skip neg id
                if inst_id < 0:
                    continue
                for t_obj in data[t]:
                    t_inst_id = t_obj['inst_id']
                    # skip neg id
                    if t_inst_id < 0:
                        continue
                    if inst_id != t_inst_id:
                        iou = result.pop(0)
                        if iou > 0:
                            # tuple: (iou_value, at time t, with object inst)
                            c_obj['iou'].append((iou, t, t_inst_id))
                            t_obj['iou'].append((iou, current_tid, inst_id))
    # area 1-order calc
    last_observe = {}
    largest_area = {}
    continuous_group = {}
    current_continuous = {}
    for t in tids:
        for obj_idx in range(len(data[t])):
            obj = data[t][obj_idx]
            inst_id = obj['inst_id']
            # skip neg obj
            if inst_id < 0:
                obj['area_1order'] = 0
                obj['largest_area'] = 0
                continue
            # init observe
            if not inst_id in last_observe.keys():
                # observe: (tid, object_idx, mask_area, last_1-order_diff)
                last_observe[inst_id] = (t, obj_idx, int(obj['mask'].sum()), 0)
                largest_area[inst_id] = int(obj['mask'].sum())
                # continuous statistics: calc distance from obj to its breakpoint
                continuous_group[inst_id] = []
                current_continuous[inst_id] = [t]
                continue
            # calc 1-order area
            current_area = int(obj['mask'].sum())
            last_info = last_observe[inst_id]
            t_sep = t - last_info[0]
            # t_sep > 1 means breakpoint
            if t_sep > 1:
                continuous_group[inst_id].append(current_continuous[inst_id])
                current_continuous[inst_id] = [t]
            else:
                current_continuous[inst_id].append(t)
            area_var = current_area - last_info[2]
            # add key area_1order
            order1_diff = area_var / t_sep
            data[last_info[0]][last_info[1]]['area_1order'] = order1_diff
            current_info = (t, obj_idx, current_area, order1_diff)
            # update observe
            last_observe[inst_id] = current_info
            # update largest_area
            largest_area[inst_id] = max(largest_area[inst_id], current_info[2])
    # keep var of last id object
    for inst_id in last_observe:
        info = last_observe[inst_id]
        data[info[0]][info[1]]['area_1order'] = info[2]
        continuous_group[inst_id].append(current_continuous[inst_id])

    # write largest area and distance between breakpoint
    for t in tids:
        for obj in data[t]:
            inst_id = obj['inst_id']
            if inst_id in largest_area:
                obj['largest_area'] = largest_area[inst_id]
                asigned = False
                for time_group in continuous_group[inst_id]:
                    if t in time_group:
                        cur_idx = time_group.index(t)
                        dist = min(cur_idx + 1, len(time_group) - cur_idx)
                        obj['breakpoint_dist'] = dist
                        asigned = True
                        break
                assert asigned, 'cannot assign breakpoint dist to frame: %d id: %d' % (t, inst_id)
            else:
                obj['largest_area'] = 0
                obj['breakpoint_dist'] = -1
    
    # save data
    for f in seq_data:
        seq, tid = f.split('.')[0].split('_')
        pkl_data = data[int(tid)]
        pickle.dump(pkl_data, open(os.path.join(target_dir, f), 'wb'))

    # print('Seq %s cost: %.4fs' % (seq, time.time() - tup))

def main():
    fs = os.listdir(target_dir)
    seqs = []
    seqs_data = {}
    for f in fs:
        seq = f.split('_')[0]
        if not seq in seqs:
            seqs.append(seq)
            seqs_data[seq] = []
        seqs_data[seq].append(f)
    tup = time.time()
    for s in seqs:
        process_seq(seqs_data[s])
    print('Total cost: %.4fs' % (time.time() - tup))

if __name__ == '__main__':
    main()
