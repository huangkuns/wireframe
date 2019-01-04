import torch.utils.data as data
import numpy as np
import cv2

import os
from pathlib import Path

import pickle

from .utils import *
# from .libs import *
import ref
home = Path.home()

class JunctionTest(data.Dataset):
    def __init__(self, H, split='test', size_info=False):
        self.split = split
        self.filelst = ref.data_root / 'v1.1/test.txt'
        self.img_dir = ref.data_root / 'v1.1/test'
        
        image_size = H['image_size']
        self.image_size = (image_size, image_size)
        self.inp_h = self.inp_w = image_size

        self.size_info = size_info

        with open(self.filelst, 'r') as f:
            filenames = f.read().splitlines()
            self.imgnames = [t for t in filenames if Path.is_file(self.img_dir.joinpath(t))]
        
        if self.size_info:
            self.images = [cv2.imread(join(self.img_dir, x)) for x in self.imgnames]
            self.img_sizes = [np.array(x.shape[:2], dtype=np.int32) for x in self.images]
        else:
            self.img_sizes = None

        print(("number of Test images: {}".format(len(self.imgnames))))
        self.nImages = len(self.imgnames)

    def __getitem__(self, index):
        imgname = self.imgnames[index]
        image = self.images[index] - ref.pixel_mean
        image = cv2.resize(image, (self.inp_w, self.inp_h))
        image = image.transpose((2, 0, 1))
        
        if self.size_info:
            img_size = self.img_sizes[index]
            return image, imgname[:-4], img_size
        else:
            return image, imgname[:-4] 

    def __len__(self):
        return self.nImages

class Junction(data.Dataset):
    def __init__(self, H, split='train', debug=False):
        H = H
        self.split = split
        self.debug = debug
        
        self.grid_h = self.grid_w = H['grid_size']
        self.num_grids = self.grid_w * self.grid_h
        self.max_len = H["max_len"]
        self.inp_w = self.inp_h = H['image_size']
        self.num_bin = H['num_bin']

        self.balance = H.get('balance', False)
        self.ratio = H.get('ratio', 100000)

        suffix = "{}_{}_{}".format(self.inp_w, self.grid_w, self.num_bin)
        self.ext = ref.ext
        
        if split == 'val':
            self.src_dir = save_dir / suffix
            self.filelst = save_dir / "{}_val.txt".format(suffix)
        else:
            print("Reading processed data.")
            save_dir = ref.junc_data_root / 'processed'
            self.src_dir = save_dir / suffix
            trainlst = H.get('trainlst', '')
            self.filelst = trainlst if trainlst else save_dir / "{}_train.txt".format(suffix) 
            print("train list file {}".format(self.filelst))
        
        with open(self.filelst, 'r') as f:
            self.imgnames = f.read().splitlines()
        self.nImages = len(self.imgnames)
        print("{} split: {}".format(split, self.nImages))
        self.img_size = np.array((self.inp_h, self.inp_w), dtype=np.int32)

        print("{}, ratio: {}".format('sample' if self.balance else 'not sample', 
                                    self.ratio if self.balance else 'None'))

    def __getitem__(self, index):
        imgname = self.imgnames[index]
        with open(self.src_dir.joinpath(imgname[:-4] + self.ext), 'rb') as handle:
            output = pickle.load(handle, encoding='latin1')
            junc_conf = output['junction_flags'][:, :, 0]
            junc_res = output['junction'][:, :, 0, :].transpose((2, 0, 1))
            bin_conf = output['theta_bin_conf'][:, :, 0,  :].transpose((2, 0, 1))
            bin_res = output['theta_bin_residual'][:, :, 0, :].transpose((2, 0, 1))
            if self.balance:
                junc_conf = sample_(junc_conf, self.ratio)

            image = output['image'] - ref.pixel_mean
            image = image.transpose((2, 0, 1))
            imgname = output['imgname']

            return image, junc_conf, junc_res, bin_conf, bin_res, imgname, self.img_size

    def __len__(self):
        return self.nImages

def sample_(label, ratio):
    ## select some neg sample and assign lablel 2
    ## In loss function, add class weights 1., 1., 0. for class 0, 1, 2
    nlabel = 2 - label
    neg_indexes = np.where(label == 0)
    pos_indexes = np.where(label == 1)
    num_neg = neg_indexes[0].shape[0]
    num_pos = pos_indexes[0].shape[0]
    num_neg_keep = min(num_neg, ratio * num_pos)
    select_indexes = np.random.choice(np.arange(num_neg), int(num_neg_keep))
    select_neg_indexes = (neg_indexes[0][select_indexes], neg_indexes[1][select_indexes])
    nlabel[select_neg_indexes] = 0
    # ignore_num = np.where(nlabel == 2)[0].shape[0]

    return nlabel

if __name__ == "__main__":
    main()

