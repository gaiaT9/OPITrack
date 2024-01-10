"""
Author: Yan Gao
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
from numpy.core.numeric import identity
from datasets.mots_tools.mots_eval.eval import filename_to_frame_nr
import os, sys
import shutil
import time
from config import *
import prettytable

os.chdir(rootDir)

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from config_mots import *
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer
from file_utils import remove_key_word
import subprocess

from test_tracking_fast import MOTSEvaluator, validate_mots

torch.backends.cudnn.benchmark = True
config_name = sys.argv[1]

import logging
log_dir = os.path.join('exps', config_name)
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log'), 'w'), logging.StreamHandler()])

args = eval(config_name).get_args()
if 'cudnn' in args.keys():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args['save_dir'] = os.path.join('exps', config_name, args['save_dir'])
if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# train dataloader
train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
logging.info(args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# train_dataset_it = torch.utils.data.DataLoader(
#    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
#    num_workers=0, pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])

# set device
device = torch.device("cuda" if args['cuda'] else "cpu")
model = torch.nn.DataParallel(model).to(device)
# model = torch.nn.DataParallel(model).cuda()

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=0.1)

# clustering
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# MOTS Evaluator
seq_map = os.path.join(rootDir, "datasets/mots_tools/mots_eval", "val.seqmap")
kitti_instance_root = os.path.join(kittiRoot, 'instances')
evaluator = MOTSEvaluator(kitti_instance_root, seq_map)

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# resume
start_epoch = 1
best_iou = 0
best_hota = 0
best_epoch = 0
best_hota_epoch = 0
best_seed = 10
if 'resume_path' in args.keys() and args['resume_path'] is not None and os.path.exists(args['resume_path']):
    logging.info('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    if 'start_epoch' in args.keys():
        start_epoch = args['start_epoch']
    elif 'epoch' in state.keys():
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 1
    # best_iou = state['best_iou']
    for kk in state.keys():
        if 'state_dict' in kk:
            state_dict_key = kk
            break
    new_state_dict = state[state_dict_key]
    if not 'state_dict_keywords' in args.keys():
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except:
            logging.info('resume checkpoint with strict False')
            model.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = remove_key_word(state[state_dict_key], args['state_dict_keywords'])
        model.load_state_dict(new_state_dict, strict=False)
    try:
        logger.data = state['logger_data']
    except:
        pass


def train(epoch):
    # define meters
    loss_meter = AverageMeter()
    loss_emb_meter = AverageMeter()

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        logging.info('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):

        points = sample['points']
        xyxys = sample['xyxys']
        labels = sample['labels']
        emb_loss = model(points, labels, xyxys)

        loss = emb_loss.mean()
        if loss.item() > 0:
            loss_emb_meter.update(emb_loss.mean().item())
            loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_meter.avg, loss_emb_meter.avg


def val(epoch):
    state = {
        'epoch': epoch,
        'best_iou': best_iou,
        'best_seed': best_seed,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'logger_data': logger.data
    }
    file_name = os.path.join(args['save_dir'], 'checkpoint.pth')
    torch.save(state, file_name)

    val_name = args['eval_config']
    # val_args = eval(val_name).get_args()
    # runfile = "test_tracking.py"
    # p = subprocess.run([pythonPath, "-u", runfile,
    #                     val_name], stdout=subprocess.PIPE, cwd=rootDir)

    val_args = eval(val_name).get_args()
    save_val_dir = val_args['save_dir'].split('/')[1]

    val_args['run_eval'] = False
    validate_mots(val_args, verbose=False, no_cache=True, specific_checkpoints_name=file_name)
    # p = subprocess.run([pythonPath, "-u", "eval.py",
    #                     os.path.join(rootDir, save_val_dir), kittiRoot + "/instances", "val.seqmap"],
    #                    stdout=subprocess.PIPE, cwd=rootDir + "/datasets/mots_tools/mots_eval")
    # pout = p.stdout.decode("utf-8")
    class SymmaryLogger:
        def __init__(self):
            self.out_str = ""
        def reset(self):
            self.out_str = ""
        def log(self, s):
            self.out_str += (s + '\n')
    
    summary = SymmaryLogger()
    
    evaluator.evaluate(os.path.join(rootDir, save_val_dir), logger=summary.log)
    pout = summary.out_str

    # HOTA metric calc
    p = subprocess.run([pythonPath, "-u", os.path.join(rootDir, 'datasets', 'TrackEval', 'scripts', 'run_kitti_mots.py'),
                        '--GT_FOLDER', os.path.join(kittiRoot, 'training'), '--TRACKERS_FOLDER', '..', '--SEQMAP_FILE',
                        os.path.join(rootDir, 'datasets', 'mots_tools', 'mots_eval', 'val.seqmap'), '--TRACKER_SUB_FOLDER',
                        save_val_dir, '--TRACKERS_TO_EVAL', 'PointTrack'],
                       stdout=subprocess.PIPE, cwd=rootDir)
    hotaout = p.stdout.decode("utf-8")

    if 'person' in args['save_dir']:
        class_str = "Evaluate class: Pedestrians"
        hota_str = 'PointTrack-pedestrian'
    else:
        class_str = "Evaluate class: Cars"
        hota_str = 'PointTrack-car'
    # parse raw CLEAR metric
    pout = pout[pout.find(class_str):]
    # print result
    print_metric = ['sMOTSA', 'MOTSA', 'MOTSP', 'MOTSAL', 'MODSA', 'MODSP', 'Recall', 'Prec', 
                    'F1', ' FAR', 'MT', 'PT', 'ML', 'TP', 'FP', 'FN', 'IDS', 'Frag', 'GT Obj',
                    'GT Trk', 'TR Obj', 'TR Trk', 'Ig TR Tck']
    print_value = pout[pout.find('all   '):].split('\n')[0].strip().split(' ')
    print_value.remove('all')
    while True:
        if '' in print_value:
            print_value.remove('')
        else:
            break
    tb = prettytable.PrettyTable(print_metric)
    tb.add_row(print_value)
    logging.info(tb)
    # print(pout[pout.find('all   '):][6:126].strip())
    acc = pout[pout.find('all   '):][6:26].strip().split(' ')[0]

    # parse HOTA metric output
    hota_elements = hotaout.split('\n\n')
    valid_hota_result = []
    for element in hota_elements:
        if element.split()[1] == hota_str:
            valid_hota_result.append(element)

    hota_result, clear_result, identity_result, count_result = valid_hota_result

    hota_keys = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'RHOTA', 'HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
    clear_keys = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'CLR_TP', 'CLR_FN', 'CLR_FP',
                  'IDSW', 'MT', 'PT', 'ML', 'Frag']
    identity_keys = ['IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP']
    count_keys = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']

    combined_hota_result = hota_result.split('\n')[-1]
    combined_clear_result = clear_result.split('\n')[-1]
    combined_identity_result = identity_result.split('\n')[-1]
    # when eval pedstrian, last row is empty line
    split_count_result = count_result.split('\n')
    if split_count_result[-1] != '':
        combined_count_result = split_count_result[-1]
    else:
        combined_count_result = split_count_result[-2]

    hota_value = combined_hota_result.split()[1:]
    clear_value = combined_clear_result.split()[1:]
    identity_value = combined_identity_result.split()[1:]
    count_value = combined_count_result.split()[1:]

    logging.info('HOTA metrics:')
    tb = prettytable.PrettyTable(hota_keys)
    tb.add_row(hota_value)
    logging.info(tb)

    logging.info('New CLEAR metric:')
    tb = prettytable.PrettyTable(clear_keys)
    tb.add_row(clear_value)
    logging.info(tb)

    logging.info('ID metrics:')
    tb = prettytable.PrettyTable(identity_keys)
    tb.add_row(identity_value)
    logging.info(tb)

    logging.info('Count metrics:')
    tb = prettytable.PrettyTable(count_keys)
    tb.add_row(count_value)
    logging.info(tb)

    return 0.0, float(acc), float(hota_value[0])


def save_checkpoint(state, is_best, iou_str, is_best_hota, hota_str, is_lowest=False, name='checkpoint.pth'):
    logging.info('=> saving checkpoint')
    if 'save_name' in args.keys():
        file_name = os.path.join(args['save_dir'], args['save_name'])
    else:
        file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth' + iou_str))
    if is_lowest:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_seed_model.pth'))
    if is_best_hota:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_hota_model.pth' + hota_str))

# val first
logging.info('Validating on initialization')
val_loss, val_iou, val_hota = val(0)
logging.info('===> val loss: {:.4f}, val iou: {:.4f}, val HOTA: {:.4f}'.format(val_loss, val_iou, val_hota))
logger.add('val', val_loss)
logger.add('iou', val_iou)

for epoch in range(start_epoch, args['n_epochs']):
    logging.info('Starting epoch {}'.format(epoch))
    # scheduler.step(epoch)
    # if epoch == start_epoch:
    #     print('Initial eval')
    #     val_loss, val_iou = val(epoch)
    #     print('===> val loss: {:.4f}, val iou: {:.4f}'.format(val_loss, val_iou))

    train_loss, emb_loss = train(epoch)
    logging.info('===> train loss: {:.4f}, train emb loss: {:.4f}'.format(train_loss, emb_loss))
    logger.add('train', train_loss)

    if 'val_interval' not in args.keys() or epoch % args['val_interval'] == 0:
        val_loss, val_iou, val_hota = val(epoch)
        logging.info('===> val loss: {:.4f}, val iou: {:.4f}, val HOTA: {:.4f}'.format(val_loss, val_iou, val_hota))
        logger.add('val', val_loss)
        logger.add('iou', val_iou)
        # logger.plot(save=args['save'], save_dir=args['save_dir'])

        # sMOTSA
        is_best = val_iou > best_iou
        if is_best:
            best_epoch = epoch
        best_iou = max(val_iou, best_iou)

        # HOTA
        is_best_hota = val_hota > best_hota
        if is_best_hota:
            best_hota_epoch = epoch
        best_hota = max(val_hota, best_hota)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_iou': best_iou,
                'best_seed': best_seed,
                'best_hota': best_hota,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data
            }
            for param_group in optimizer.param_groups:
                lrC = str(param_group['lr'])
            save_checkpoint(state, is_best, str(best_iou) + '_' + lrC, is_best_hota, str(best_hota) + '_' + lrC, is_lowest=False)

logging.info('best sMOTSA %.4f, at epoch %d' % (best_iou, best_epoch))
logging.info('best HOTA %.4f, at epoch %d' % (best_hota, best_hota_epoch))