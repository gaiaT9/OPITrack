import os, sys
import numpy as np
import pickle
from tqdm import tqdm
import time

target_dir = sys.argv[1]
# target_dir = 'dataImgPredTrackEnvDB'
# target_dir = 'dataMOTChallengePredTrackEnvDB/'

# configs
# calc IoU and record >0 samples
# mark segm area drastic changed, use 1-order diff to measure
# iou_window = -1 means global iou calculations
tmp_dir = '/dev/shm/'
tmp_prefix = 'iou_calc_'
iou_window = 100

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

def process_seq(tup):
    t_sep, seq_data = tup
    iou_data = {}
    data = {}
    seq = seq_data[0].split('.')[0].split('_')[0]
    except_file_path = os.path.join(tmp_dir, tmp_prefix + '%s_%d.pkl'%(seq, t_sep))
    # if file already exists skip it
    if os.path.exists(except_file_path):
        return
    # load data
    for f in seq_data:
        tid = int(f.split('.')[0].split('_')[1])
        pkl = pickle.load(open(os.path.join(target_dir, f), 'rb'))
        iou_data[tid] = [[] for i in range(len(pkl))]
        data[tid] = pkl
        # data[tid] = lambda: pickle.load(open(os.path.join(target_dir, f), 'rb'))
    tids = list(data.keys())
    tids.sort()
    # iou calculations
    for i in tqdm(range(len(tids) - 1)):
        current_tid = tids[i]
        if not current_tid in tids:
            continue
        # t sep
        t = current_tid + t_sep
        if not t in tids:
            continue
        current_data = data[current_tid]
        t_data = data[t]
        for c_idx in range(len(current_data)):
            c_obj = current_data[c_idx]
            inst_id = c_obj['inst_id']
            # skip neg id
            if inst_id < 0:
                continue
            for t_idx in range(len(t_data)):
                t_obj = t_data[t_idx]
                t_inst_id = t_obj['inst_id']
                # skip neg id
                if t_inst_id < 0:
                    continue
                if inst_id != t_inst_id:
                    iou = calc_iou(c_obj, t_obj)
                    if iou > 0:
                        # tuple: (iou_value, at time t, with object inst)
                        iou_data[current_tid][c_idx].append((iou, t, t_inst_id))
                        iou_data[t][t_idx].append((iou, current_tid, inst_id))
        del data[current_tid]
    
    # save data
    pickle.dump(iou_data, open(except_file_path, 'wb'))

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
    import multiprocessing as mp
    pool = mp.Pool(processes=16)
    tup = time.time()
    # make tmp iou pickle
    print('calculating...')
    payload = []
    for s in seqs:
        for t in range(iou_window):
            payload.append((t + 1, seqs_data[s]))
        # process_seq(seqs_data[s])
    pool.map(process_seq, payload)
    pool.close()
    print('calculating done!')
    # merge result
    for s in seqs:
        print('merging %s...' % s)
        seq_iou_result = {}
        for t_sep in range(1, iou_window + 1):
            sep_result = pickle.load(open(os.path.join(tmp_dir, tmp_prefix + '%s_%d.pkl'%(s, t_sep)), 'rb'))
            for t in sep_result.keys():
                if not t in seq_iou_result:
                    seq_iou_result[t] = [[] for _ in range(len(sep_result[t]))]
                for i in range(len(sep_result[t])):
                    seq_iou_result[t][i].extend(sep_result[t][i])
        # write result
        for f in seqs_data[s]:
            target_path = os.path.join(target_dir, f)
            tid = int(f.split('.')[0].split('_')[1])
            pkl = pickle.load(open(target_path, 'rb'))
            for i in range(len(pkl)):
                pkl[i]['iou'] = seq_iou_result[tid][i]
            pickle.dump(pkl, open(target_path, 'wb'))

    print('Total cost: %.4fs' % (time.time() - tup))

if __name__ == '__main__':
    main()
