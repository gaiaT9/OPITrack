import os, sys
import numpy as np
import pickle
from tqdm import tqdm
import time

target_dir = sys.argv[1]

raw_keys = {'inst_id', 'sp', 'img', 'mask', 'maskX'}

def process_data(f):
    pkl = pickle.load(open(os.path.join(target_dir, f), 'rb'))
    new_pkl = []
    for o in pkl:
        new_dict = {}
        for k in raw_keys:
            new_dict[k] = o[k]
        new_pkl.append(new_dict)
    pickle.dump(new_pkl, open(os.path.join(target_dir, f), 'wb'))

def process_seq(seq_data):
    for f in tqdm(seq_data):
        process_data(f)

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
    # payload = []
    tup = time.time()
    for s in seqs:
        # payload.append(seqs_data[s])
        process_seq(seqs_data[s])
    import multiprocessing as mp
    # pool = mp.Pool(processes=len(seqs))
    # pool.map(process_seq, payload)
    print('Total cost: %.4fs' % (time.time() - tup))

if __name__ == '__main__':
    main()
