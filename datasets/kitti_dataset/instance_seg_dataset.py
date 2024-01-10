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


class MOTSTest(Dataset):
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}

    def __init__(self, root_dir='./', type="train", class_id=26, size=None, transform=None, batch=False, batch_num=8):

        print('Kitti Dataset created')
        self.batch = batch
        self.batch_num = batch_num

        self.class_id = class_id
        self.size = size
        self.transform = transform

        self.mots_image_root = os.path.join(kittiRoot, 'testing/image_02')
        self.timestamps = self.TIMESTEPS_PER_SEQ_TEST

        self.mots_car_pairs = []
        for subdir in self.SEQ_IDS_TEST:
            image_list = sorted(make_dataset(os.path.join(self.mots_image_root, subdir), suffix='.png'))
            image_list = ['/'.join(el.split('/')[-2:]) for el in image_list]
            self.mots_car_pairs += image_list

        self.mots_num = len(self.mots_car_pairs)
        self.mots_class_id = 2

    def __len__(self):

        return self.mots_num if self.size is None else self.size

    def get_data_from_mots(self, index):
        # random select and image and the next one
        # index = random.randint(0, self.mots_num - 1)
        path = self.mots_car_pairs[index]
        sample = {}
        image = Image.open(os.path.join(self.mots_image_root, path))

        sample['mot_image'] = image
        sample['mot_im_name'] = path
        sample['im_shape'] = image.size

        return sample

    def __getitem__(self, index):
        sample = self.get_data_from_mots(index)

        # transform
        if(self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


class MOTSCars(Dataset):

    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}

    def __init__(self, root_dir='./', type="train", size=None, transform=None, kins=True, kins_only=False):

        print('Kitti Dataset created')
        self.class_id = 26
        self.type = type

        if type == 'crop':
            self.image_list = make_dataset(os.path.join(kittiRoot,'crop_KINS', 'images'), suffix='.png')
            self.instance_list = [el.replace('images', 'instances') for el in self.image_list]
        else:
            type = 'training' if type in 'training' else 'testing'
            if kins and type == 'training':
                self.image_index = self._load_image_set_index_new('training') + self._load_image_set_index_new('testing')
                self.clean_kins_inst_file = os.path.join(root_dir, 'KINSCarValid.pkl')
                if not os.path.isfile(self.clean_kins_inst_file):
                    self.instance_list = make_dataset(os.path.join(root_dir, 'training/KINS/'), suffix='.png') + make_dataset(os.path.join(root_dir, 'testing/KINS/'), suffix='.png')
                    self.instance_list = leave_needed(self.instance_list, self.class_id) # 14908 -> 13997
                    save_pickle2(self.clean_kins_inst_file, self.instance_list)
                else:
                    self.instance_list = load_pickle(self.clean_kins_inst_file)
                # get image and instance list
                image_list = [el.replace('KINS', 'image_2') for el in self.instance_list]
                self.image_list = image_list
            else:
                self.instance_list, self.image_list = [], []

            if not kins_only:
                self.mots_instance_root = os.path.join(kittiRoot, 'instances')
                self.mots_image_root = os.path.join(kittiRoot, 'images')
                assert os.path.isfile(os.path.join(kittiRoot, 'motsCarsTrain.pkl'))

                if type == 'training':
                    self.mots_persons = load_pickle(os.path.join(kittiRoot, 'motsCarsTrain.pkl'))
                    self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in self.mots_persons]
                    self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]
                    self.image_list += self.mots_image_list
                    self.instance_list += self.mots_instance_list
                else:
                    self.mots_persons = load_pickle(os.path.join(kittiRoot, 'motsCarsTest.pkl'))
                    self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in self.mots_persons]
                    self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]
                    self.image_list = self.mots_image_list
                    self.instance_list = self.mots_instance_list
        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform

    def _load_image_set_index_new(self, type):
        if type == 'training':
            train_set_file = open(rootDir + 'datasets/splits/train.txt', 'r')
            image_index = train_set_file.read().split('\n')
        else:
            val_set_file = open(rootDir + 'datasets/splits/val.txt', 'r')
            image_index = val_set_file.read().split('\n')
        return image_index

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def get_data_from_kins(self, index):
        if self.type == 'crop':
            index = random.randint(0, self.real_size - 1)
        sample = {}
        image = Image.open(self.image_list[index])
        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = self.decode_instance(instance, self.instance_list[index])  # get semantic map and instance map
        sample['image'] = image
        sample['im_name'] = self.image_list[index]
        sample['instance'] = instance
        sample['label'] = label

        return sample

    def __getitem__(self, index):
        # select two images from kins
        while 1:
            try:
                sample = self.get_data_from_kins(index)

                # transform
                if (self.transform is not None):
                    sample = self.transform(sample)
                    return sample
                else:
                    return sample
            except:
                pass

    def decode_instance(self, pic, path):
        if self.type == 'crop':
            class_id = 1 if 'MOTS' in path else 26
        else:
            class_id = 26 if 'KINS' in path else 1
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
        if self.type=='crop':
            assert mask.sum() > 0
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) & mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)

class MOTSPerson(Dataset):

    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}

    def __init__(self, root_dir='./', type="train", size=None, transform=None, kins=True, kins_only=False):

        print('Kitti Dataset created')
        self.class_id = 2
        self.type = type

        if type == 'crop':
            self.image_list = make_dataset(os.path.join(kittiRoot,'crop_person_KINS', 'images'), suffix='.png')
            self.instance_list = [el.replace('images', 'instances') for el in self.image_list]
        else:
            type = 'training' if type in 'training' else 'testing'
            if kins and type == 'training':
                self.image_index = self._load_image_set_index_new('training') + self._load_image_set_index_new('testing')
                self.clean_kins_inst_file = os.path.join(root_dir, 'KINSPersonValid.pkl')
                if not os.path.isfile(self.clean_kins_inst_file):
                    self.instance_list = make_dataset(os.path.join(root_dir, 'training/person_KINS/'), suffix='.png') + make_dataset(os.path.join(root_dir, 'testing/person_KINS/'), suffix='.png')
                    self.instance_list = leave_needed(self.instance_list, self.class_id) # 14908 -> 13997
                    save_pickle2(self.clean_kins_inst_file, self.instance_list)
                else:
                    self.instance_list = load_pickle(self.clean_kins_inst_file)
                # get image and instance list
                image_list = [el.replace('person_KINS', 'image_2') for el in self.instance_list]
                self.image_list = image_list
            else:
                self.instance_list, self.image_list = [], []

            if not kins_only:
                self.mots_instance_root = os.path.join(kittiRoot, 'instances')
                self.mots_image_root = os.path.join(kittiRoot, 'images')
                assert os.path.isfile(os.path.join(kittiRoot, 'motsPersonTrain.pkl'))

                if type == 'training':
                    self.mots_persons = load_pickle(os.path.join(kittiRoot, 'motsPersonTrain.pkl'))
                    self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in self.mots_persons]
                    self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]
                    self.image_list += self.mots_image_list
                    self.instance_list += self.mots_instance_list
                else:
                    self.mots_persons = load_pickle(os.path.join(kittiRoot, 'motsPersonTest.pkl'))
                    self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in self.mots_persons]
                    self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]
                    self.image_list = self.mots_image_list
                    self.instance_list = self.mots_instance_list
        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform

    def _load_image_set_index_new(self, type):
        if type == 'training':
            train_set_file = open(rootDir + 'datasets/splits/train.txt', 'r')
            image_index = train_set_file.read().split('\n')
        else:
            val_set_file = open(rootDir + 'datasets/splits/val.txt', 'r')
            image_index = val_set_file.read().split('\n')
        return image_index

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def get_data_from_kins(self, index):
        if self.type == 'crop':
            index = random.randint(0, self.real_size - 1)
        sample = {}
        image = Image.open(self.image_list[index])
        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = self.decode_instance(instance, self.instance_list[index])  # get semantic map and instance map
        sample['image'] = image
        sample['im_name'] = self.image_list[index]
        sample['instance'] = instance
        sample['label'] = label

        return sample

    def __getitem__(self, index):
        # select two images from kins
        while 1:
            try:
                sample = self.get_data_from_kins(index)

                # transform
                if (self.transform is not None):
                    sample = self.transform(sample)
                    return sample
                else:
                    return sample
            except:
                pass

    def decode_instance(self, pic, path):
        if self.type == 'crop':
            class_id = 2 if 'MOTS' in path else 2
        else:
            class_id = 2 if 'KINS' in path else 2
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
        if self.type=='crop':
            assert mask.sum() > 0
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) & mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)

class MOTSCarsVal(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    def __init__(self, root_dir='./', type="train", class_id=26, size=None, transform=None, batch=False, batch_num=8):

        print('Kitti Dataset created')
        type = 'training' if type in 'training' else 'testing'
        self.type = type
        self.sequence = self.SEQ_IDS_TRAIN if type == 'training' else self.SEQ_IDS_VAL
        self.batch = batch
        self.batch_num = batch_num

        self.class_id = class_id
        self.size = size
        self.transform = transform

        self.mots_instance_root = os.path.join(kittiRoot, 'instances')
        self.mots_image_root = os.path.join(kittiRoot, 'images')

        self.mots_car_pairs = []
        for subdir in self.sequence:
            instance_list = sorted(make_dataset(os.path.join(self.mots_instance_root, subdir), suffix='.png'))
            instance_list = ['/'.join(el.split('/')[-2:]) for el in instance_list]
            for i in instance_list:
                self.mots_car_pairs.append(i)

        self.mots_num = len(self.mots_car_pairs)
        self.mots_class_id = 1

    def __len__(self):

        return self.mots_num if self.size is None else self.size

    def get_data_from_mots(self, index):
        path = self.mots_car_pairs[index]
        sample = {}
        image = Image.open(os.path.join(self.mots_image_root, path))
        # load instances
        instance = Image.open(os.path.join(self.mots_instance_root, path))
        instance, label = self.decode_mots(instance, self.mots_class_id)  # get semantic map and instance map
        sample['mot_image'] = image
        sample['mot_instance'] = instance
        sample['mot_label'] = label
        sample['mot_im_name'] = path
        sample['im_shape'] = image.size

        return sample

    def __getitem__(self, index):
        sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample

    @classmethod
    def decode_mots(cls, pic, class_id=None):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if class_id is not None:
            mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
            if mask.sum() > 0:
                # keep the instance ids for tracking
                # +1 because 1000 is also a valid car, inst id >= 1
                instance_map[mask] = (pic[mask] + 1) % 1000
                class_map[mask] = 1

            # assign dontcare area to -2
            mask_others = pic == 10000
            if mask_others.sum() > 0:
                class_map[mask_others] = -2
        else:
            for i, c in enumerate(cls.class_ids):
                mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
                if mask.sum() > 0:
                    ids, _, _ = relabel_sequential(pic[mask])
                    instance_map[mask] = ids + np.amax(instance_map)
                    class_map[mask] = i + 1

        return Image.fromarray(instance_map), Image.fromarray(class_map)

class MOTSPersonVal(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    def __init__(self, root_dir='./', type="train", class_id=2, size=None, transform=None, batch=False, batch_num=8):

        print('Kitti Dataset created')
        type = 'training' if type in 'training' else 'testing'
        self.type = type
        self.sequence = self.SEQ_IDS_TRAIN if type == 'training' else self.SEQ_IDS_VAL
        self.batch = batch
        self.batch_num = batch_num

        self.class_id = class_id
        self.size = size
        self.transform = transform

        self.mots_instance_root = os.path.join(kittiRoot, 'instances')
        self.mots_image_root = os.path.join(kittiRoot, 'images')

        self.mots_car_pairs = []
        for subdir in self.sequence:
            instance_list = sorted(make_dataset(os.path.join(self.mots_instance_root, subdir), suffix='.png'))
            instance_list = ['/'.join(el.split('/')[-2:]) for el in instance_list]
            for i in instance_list:
                self.mots_car_pairs.append(i)

        self.mots_num = len(self.mots_car_pairs)
        self.mots_class_id = 1

    def __len__(self):

        return self.mots_num if self.size is None else self.size

    def get_data_from_mots(self, index):
        path = self.mots_car_pairs[index]
        sample = {}
        image = Image.open(os.path.join(self.mots_image_root, path))
        # load instances
        instance = Image.open(os.path.join(self.mots_instance_root, path))
        instance, label = self.decode_mots(instance, self.mots_class_id)  # get semantic map and instance map
        sample['mot_image'] = image
        sample['mot_instance'] = instance
        sample['mot_label'] = label
        sample['mot_im_name'] = path
        sample['im_shape'] = image.size

        return sample

    def __getitem__(self, index):
        sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample

    @classmethod
    def decode_mots(cls, pic, class_id=None):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if class_id is not None:
            mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
            if mask.sum() > 0:
                # keep the instance ids for tracking
                # +1 because 1000 is also a valid car, inst id >= 1
                instance_map[mask] = (pic[mask] + 1) % 1000
                class_map[mask] = 1

            # assign dontcare area to -2
            mask_others = pic == 10000
            if mask_others.sum() > 0:
                class_map[mask_others] = -2
        else:
            for i, c in enumerate(cls.class_ids):
                mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
                if mask.sum() > 0:
                    ids, _, _ = relabel_sequential(pic[mask])
                    instance_map[mask] = ids + np.amax(instance_map)
                    class_map[mask] = i + 1

        return Image.fromarray(instance_map), Image.fromarray(class_map)
