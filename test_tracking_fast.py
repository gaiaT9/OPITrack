"""
Author: Yan Gao
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import time
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from config_mots import *
from datasets import get_dataset
from models import get_model
from utils.mots_util import *
from config import *
import subprocess
import pickle

# import MOTS toolset
from datasets.mots_tools.mots_eval.eval import load_seqmap, load_sequences, evaluate_class

class MOTSEvaluator(object):
    def __init__(self, gt_folder, seqmap_filename, tmp_dir='./'):
        tmp_seq_path = os.path.join(tmp_dir, 'gt_seq_cache.pkl')
        if os.path.exists(tmp_seq_path):
            gt_tmp_obj = pickle.load(open(tmp_seq_path, 'rb'))
            self.seqmap = gt_tmp_obj['seqmap']
            self.max_frames = gt_tmp_obj['max_frames']
            self.gt = gt_tmp_obj['gt']
        else:
            self.seqmap, self.max_frames = load_seqmap(seqmap_filename)
            self.gt = load_sequences(gt_folder, self.seqmap)
            pickle.dump({'seqmap': self.seqmap, 'max_frames': self.max_frames, 'gt': self.gt}
                         , open(tmp_seq_path, 'wb'))

    def evaluate(self, results_folder, logger=print):
        results = load_sequences(results_folder, self.seqmap, logger=logger)

        logger("Evaluate class: Cars")
        results_cars = evaluate_class(self.gt, results, self.max_frames, 1, logger=logger)

        logger("Evaluate class: Pedestrians")
        results_ped = evaluate_class(self.gt, results, self.max_frames, 2, logger=logger)

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 0
torch.manual_seed(seed)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def main():
    config_name = sys.argv[1]
    args = eval(config_name).get_args()
    if len(sys.argv) == 3:
        no_cache = True
    else:
        no_cache = False

    validate_mots(args, no_cache=no_cache)

def validate_mots(args, verbose=True, no_cache=False, specific_checkpoints_name=None):

    max_disparity = args['max_disparity']

    if args['display']:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")

    if args['save']:
        if not os.path.exists(args['save_dir']):
            os.makedirs(args['save_dir'])

    # set device
    device = torch.device("cuda:0" if args['cuda'] else "cpu")

    # dataloader
    dataset = get_dataset(
        args['dataset']['name'], args['dataset']['kwargs'])
    dataset_it = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

    # load model
    model = get_model(args['model']['name'], args['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)

    # load snapshot and predictiction temp
    if os.path.exists(args['checkpoint_path']):
        import hashlib
        model_md5 = hashlib.md5(open(args['checkpoint_path'], 'rb').read()).hexdigest()
        tmp_pred_file = os.path.join('/dev/shm', os.path.basename(args['checkpoint_path']).split('.')[0] + model_md5[:8] + '.pkl')
        if os.path.exists(tmp_pred_file) and not no_cache:
            tmp_pred = pickle.load(open(tmp_pred_file, 'rb'))
            print('load tmp prediction from %s' % tmp_pred_file)
        else:
            tmp_pred = {}
        if specific_checkpoints_name is None:
            state = torch.load(args['checkpoint_path'])
        else:
            state = torch.load(specific_checkpoints_name)
        model.load_state_dict(state['model_state_dict'], strict=True)
        if verbose:
            print('Load dict from %s' % args['checkpoint_path'])
    else:
        # assert(False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))
        if verbose:
            print(args['checkpoint_path'])
        raise ValueError('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

    model.eval()


    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image


    dColors = [(128, 0, 0), (170, 110, 40), (128, 128, 0), (0, 128, 128), (0, 0, 128), (230, 25, 75), (245, 130, 48)
            , (255, 225, 25), (210, 245, 60), (60, 180, 75), (70, 240, 240), (0, 130, 200), (145, 30, 180), (240, 50, 230)
            , (128, 128, 128), (250, 190, 190), (255, 215, 180), (255, 250, 200), (170, 255, 195), (230, 190, 255), (255, 255, 255)]

    trackHelper = TrackHelper(args['save_dir'], model.module.margin, alive_car=30, car=args['car'] if 'car' in args.keys() else True,
                            mask_iou=True)
    # total_pred = 0
    with torch.no_grad():
        cur_subf = None
        for sample in tqdm(dataset_it):
            subf, frameCount = sample['name'][0][:-4].split('/')[-2:]

            if cur_subf is None:
                cur_subf = subf
            else:
                if cur_subf != subf:
                    cur_subf = subf
                    # eval the last frame

            pred_identity = subf + '_' + frameCount        

            frameCount = int(float(frameCount))

            # use tmp result if exist
            if pred_identity in tmp_pred:
                embeds = tmp_pred[pred_identity]['embeds']
                masks = tmp_pred[pred_identity]['masks']
            else:
                # MOTS forward with tracking
                points = sample['points']
                # total_pred += len(points)
                if len(points) < 1:
                    embeds = np.array([])
                    masks = np.array([])
                else:
                    masks = sample['masks'][0]
                    xyxys = sample['xyxys']
                    embeds = model(points=points, labels=None, xyxys=xyxys, infer=True)
                    embeds = embeds.cpu().numpy()
                    masks = masks.numpy()

                tmp_pred[pred_identity] = {'embeds': embeds, 'masks': masks}

            # do tracking
            trackHelper.tracking(subf, frameCount, embeds, masks)

        trackHelper.export_last_video()
    # print('Total pred: %d' % total_pred)
    if not no_cache:
        pickle.dump(tmp_pred, open(tmp_pred_file, 'wb'))

    if 'run_eval' in args.keys() and args['run_eval']:
        # run eval
        from datasets.mots_tools.mots_eval.eval import run_eval
        save_val_dir = args['save_dir'].split('/')[1]
        seq_map = os.path.join(rootDir, "datasets/mots_tools/mots_eval", "val.seqmap")
        kitti_instance_root = os.path.join(kittiRoot, 'instances')

        evaluator = MOTSEvaluator(kitti_instance_root, seq_map)
        evaluator.evaluate(save_val_dir)
        # run_eval(save_val_dir, kitti_instance_root, seq_map)
        # p = subprocess.run([pythonPath, "-u", "eval.py",
        #                     os.path.join(rootDir, save_val_dir), kittiRoot + "/instances", "val.seqmap"],
        #                                stdout=subprocess.PIPE, cwd=rootDir + "/datasets/mots_tools/mots_eval")
        # print(p.stdout.decode("utf-8"))

if __name__ == '__main__':
    main()
