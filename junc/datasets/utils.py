import itertools
import json
import os
import random
import sys
from os.path import isdir, isfile, join, basename

from math import cos, pi, sin, sqrt, acos 
from tqdm import tqdm
import multiprocessing

import scipy.io as sio
import numpy as np
import cv2
from cv2 import resize as resize_, imread 

from joblib import Parallel, delayed
import progressbar as pb

# python3
import pickle

# from .libs import *
# safe dictionary copy
from copy import deepcopy as dc
random.seed(0)
import ref

# transform annotation to array format
def ann_to_array(H, d, max_len=1):
    # junction_order = H.get('junction_order', 'off_cell_center')
    region_size = H['region_size']
    inp_h = inp_w = H['image_size']
    grid_h = grid_w = H['grid_size']
    assert region_size == inp_h // grid_h
    
    junctions = d['junction']
    theta = d['theta_bin']
    num_junc = len(junctions)

    junction_residual = np.zeros((grid_h, grid_w, max_len, 2), dtype=np.float)
    junction_flags = np.zeros((grid_h, grid_w, max_len), dtype=np.int32)
    theta_bin = np.zeros(
        (grid_h, grid_w, max_len, H['num_bin']), dtype=np.int32)
    theta_bin_residual = np.zeros(
        (grid_h, grid_w, max_len, H['num_bin']), dtype=np.float)
    
    focus_range = 0.5 #H['focus_size']

    for h in range(grid_h):
        for w in range(grid_w):
            ccx, ccy = w + 0.5, h + 0.5
            cell_center_w, cell_center_h = ccx * region_size, ccy * region_size
            unsorted_junctions = []
            for idx, (jx, jy) in enumerate(junctions):
                #px, py = jx / float(region_size), jy /float(region_size)
                if abs(jx - cell_center_w) <= focus_range * region_size and abs(jy - cell_center_h) <= focus_range * region_size:
                    ox, oy = jx - cell_center_w, jy - cell_center_h
                    th = theta[idx]
                    unsorted_junctions += [(ox, oy, th)]
            if len(unsorted_junctions) == 0:
                continue 
            unsorted_junctions = unsorted_junctions[:max_len]
            #print (unsorted_junctions[0][:2])
            sorted_junctions = sorted( unsorted_junctions, key=lambda x: x[0]**2 + x[1]**2)
            num_keep = min(max_len, len(sorted_junctions))
            
            for idx_sorted in range(num_keep):
                ox, oy, th = sorted_junctions[idx_sorted]
                junction_residual[h, w, idx_sorted, :] = np.array((ox, oy), dtype=np.float32)

                order_th = len(th)
                if H['num_classes'] > 2:
                    junction_flags[h, w, idx_sorted] = min(order_th - 1, 5)
                else:
                    junction_flags[h, w, idx_sorted] =  1
                for _, tt in enumerate(th):
                    bin_idx, bin_residual = tt
                    bin_idx = int(bin_idx)
                    theta_bin[h, w, idx_sorted, bin_idx] = 1
                    theta_bin_residual[h, w, idx_sorted, bin_idx] = float(bin_residual)
    output = {}
    output['junction'] = junction_residual
    output['junction_flags'] = junction_flags
    output['theta_bin_conf'] = theta_bin
    output['theta_bin_residual'] = theta_bin_residual
    return output

## Calculate the bin index and residual of junction angles.
def make_bin(bn, ths):
    bin_num = bn
    bin_width = 360. / float(bin_num)
    bin_max_width = 4
    centers = np.array(
        [i * bin_width for i in range(bin_num)] + [360.], dtype=np.float32)

    th_bin = [None for _ in ths]
    for cnt0, th in enumerate(ths):
        bin_t = [None for _ in th]
        for cnt, angle in enumerate(th):
            #assert scale <= line_max_len, "scale should be smaller than max length of lines."
            idx = round(float(angle) / bin_width)
            idx = int(idx)
            idx1 = idx if idx != bin_num else 0

            bin_t[cnt] = (idx1, angle - centers[idx])
        th_bin[cnt0] = bin_t
    return th_bin


def dist(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def lineangle(p1, p2):
    """ p1 -> p2 """
    x1, y1 = p1
    x2, y2 = p2
    dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if y1 < y2:
        theta = acos((x2 - x1) / dist)
    else:
        theta = acos((x1 - x2) / dist)
    
    return theta / pi * 180.

# Resize image and annotation to input size of network.
def resize_and_transform(H, d, max_len = 1):
    num_bin = H['num_bin']
    
    inp_h = inp_w = H['image_size']
    img, junctions, points = d['img'], d['junction'], d['points']
    cur_h, cur_w = img.shape[:2]
    if cur_h == 0 or cur_w == 0:
        print("{} {} {}"%(d['imgname'], cur_h, cur_w))
    sw, sh = float(inp_w) / float(cur_w), float(inp_h) / float(cur_h)

    try:
        resized_img = resize_(img, (inp_h, inp_w), cv2.INTER_CUBIC)
        d['junction'] = [(sw * x, sh * y) for x, y in junctions]
        d['points'] = [(sw * x, sh * y) for x, y in points]
    except ValueError:
        print("cannot resize ", d['imgname'])
        raise ValueError

    ntheta = resize_theta(d['theta'], (sw, sh))
    d['theta'] = ntheta
    d['theta_bin'] = make_bin(num_bin, ntheta)
    d['img'] = resized_img
    
    output = ann_to_array(H, d)
    output['imgname'] = d['imgname']
    output['image'] = resized_img
    
    return output

# when resizing image, the junction angles should be
# calculated again.
def resize_theta(ths, scale_param, with_conf=False):
    new_ths = [None for _ in ths]

    for cnt0, th in enumerate(ths):
        bin_t = [None for _ in th]
        if with_conf:
            for cnt, (t, conf_t) in enumerate(th):
                x = cos(t * pi / 180.) * scale_param[0]
                y = sin(t * pi / 180.) * scale_param[1]
                dist = sqrt(x**2 + y**2)
                if abs(y) <= 0.001:
                    nt = 180. if x < 0 else 0.
                elif y > 0:
                    nt = acos(x / dist) / pi * 180.
                else:
                    nt = 360. - acos(x / dist) / pi * 180.
                bin_t[cnt] = (nt, conf_t)
        else:
            for cnt, t in enumerate(th):
                x = cos(t * pi / 180.) * scale_param[0]
                y = sin(t * pi / 180.) * scale_param[1]
                dist = sqrt(x**2 + y**2)
                if abs(y) <= 0.001:
                    nt = 180. if x < 0 else 0.
                elif y > 0:
                    nt = acos(x / dist) / pi * 180.
                else:
                    nt = 360. - acos(x / dist) / pi * 180.
                bin_t[cnt] = nt
        new_ths[cnt0] = bin_t
    return new_ths


def resize_points(pts, scale_param):
    sx, sy = scale_param
    points = [None for _ in pts]
    for idx, (x, y) in enumerate(pts):
        points[idx] = (x * sx, y * sy)
    return points

###   make crop augmentation.

def make_crop(d, bbox, suffix):
    d_crop = dc(d)
    (x0, y0, x1, y1) = bbox
    I = d['img']
    junction = d['junction']
    theta = d['theta']

    d_crop['img'] = I[y0:y1, x0:x1, :]
    d_crop['imgname'] = "{}_{}.jpg".format(d['imgname'][:-4], suffix[0])

    new_junction = []
    new_theta = []
    for pp, ths in zip(junction, theta):
        x, y = pp
        if x >= x0 and x < x1 and y >= y0 and y < y1:
            new_junction += [(x - x0, y - y0)]
            new_theta += [ths]
    d_crop['points'] = [(x - x0, y - y0) for x, y in d['points']]
    d_crop['junction'] = new_junction
    d_crop['theta'] = new_theta

    return d_crop


# make mirror augmentation.
def make_mirror(d, axis=1):
    I = d['img']
    h, w = I.shape[:2]
    
    d_mirror = dc(d)
    suffix = 'mr' if axis == 2 else 'ud'
    d_mirror['imgname'] = "{}_{}.jpg".format(d['imgname'][:-4], suffix)

    if axis == 2:
        d_mirror['img'] = I[:, ::-1, :]
        d_mirror['junction'] = [(w - x, y) for x, y in d['junction']]
        d_mirror['points'] = [(w - x, y) for x, y in d['points']]
        d_mirror['theta'] = [[180. - th if th < 180. else 540. - th for th in point_th]
                              for point_th in d['theta']]

    elif axis == 1:
        d_mirror['img'] = I[::-1, :, :]
        d_mirror['junction'] = [(x, h - y) for x, y in d['junction']]
        d_mirror['points'] = [(x, h - y) for x, y in d['points']]
        d_mirror['theta'] = [[360. - th for th in point_th]
                          for point_th in d['theta']]

    return(d_mirror)


# Transform the raw data and augment, store it.
def save_ann(H, d, save_dir, max_len = 1, split='train'):
    grid_h = grid_w = H['grid_size']
    image_h = image_w  = H['image_size']
    num_bin = H['num_bin']

    I = d['img']
    if len(I.shape) < 3:
        raise
        return
    if I.shape[2] == 4:
        I = I[:, :, :3]
    
    td = dc(d)
    td['img'] = I
    td['theta'] = theta = [[ x for x, _ in th] for th in d['theta']]

    annlist = []
    annlist += [td]
    
    if split != 'test':
        annlist += [make_mirror(td, axis=1), make_mirror(td, axis=2)]
        h, w = I.shape[:2]
        junction = td['junction']
        crop_list = []
        
        if h > w:
            x0, x1 = 0, w
            
            y0, y1 = 0, w
            crop_list += [(x0, y0, x1, y1, 'top')]
            
            y0, y1 = int((h - w) / 2.), int((h-w)/ 2.) + w
            crop_list += [(x0, y0, x1, y1, 'middle')]
            
            y0, y1 = h - w, h
            crop_list += [(x0, y0, x1, y1, 'bottom')]
        elif w > h:
            y0, y1 = 0, h
            
            x0, x1 = 0, h
            crop_list += [(x0, y0, x1, y1, 'left')]
            
            x0, x1 = int((w - h) / 2.), int( (w - h)/2. ) + h
            crop_list += [(x0, y0, x1, y1, 'middle')]
            
            x0, x1 = w - h, w
            crop_list += [(x0, y0, x1, y1, 'right')]
        else:
            x0, y0 = int(h * 0.25), int(h * 0.25)
            x1, y1 = int(h * 0.75), int(h * 0.75)
            crop_list += [(x0, y0, x1, y1, 'center')]
        
        annlist += [make_crop(td, (x0, y0, x1, y1), suffix) for x0, y0, x1, y1, suffix in crop_list]

    outputs = [resize_and_transform(H, t) for t in annlist]

    for fn in outputs:
        junc_flags = fn['junction_flags']
        num_junc = np.sum(junc_flags.astype(float))
        if split != 'test' and num_junc <= 5:
            continue
        imgname = fn['imgname']
        dir_to_save = join(save_dir, "{}_{}_{}".format(image_h, grid_h, num_bin))
        if not isdir(dir_to_save):
            os.mkdir(dir_to_save)
        with open('{}/{}.pickle'.format(dir_to_save, imgname[:-4]), 'wb') as handle:
            pickle.dump(fn, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cv2.imwrite(join(dir_to_save, fn['imgname']), fn['image'])

## load the pickle files.
def loading(fn, src_dir):
    with open(src_dir / fn, 'rb') as handle:
        d = pickle.load(handle, encoding='latin1')
    return d


# Junction label augmentation cannot be easily done online.
# Precompute the label, store locally.
debug = False
def load_data_and_save(H, src_dir, save_dir, filenames, split='train', use_mp=True):
    print("src_dir: {} save_dir: {}".format(src_dir, save_dir))
    
    bar = pb.ProgressBar(widgets=[ '[ ', pb.Bar(), ' ][ ', pb.Timer(), ' ]'], max_value=pb.UnknownLength)
    def loadDataset():
        return Parallel(n_jobs=1)(delayed(loading)(f, src_dir) for f in bar(filenames))
    
    if not debug:
        print("== loading raw data ==")
        anno = loadDataset()
        total_num = len(anno)
        print("== {} raw images data loaded ==".format(total_num))
        
        cpu_num = multiprocessing.cpu_count()
        cpu_num = min(30, cpu_num)
        
        bar = pb.ProgressBar(widgets=[ '[ ', pb.Bar(), ' ][ ', pb.Timer(), ' ]'], max_value=pb.UnknownLength)
        if use_mp:
            Parallel(n_jobs=cpu_num)(delayed(save_ann)(H, d, save_dir, split=split)
                for d in bar(anno))  # max_len default to 1.
        else:
            print("== single process processing")
            for d in anno: # max_len default to 1.
                save_ann(H, d, save_dir, split=split)
    
    image_h, grid_h, num_bin = H['image_size'], H['grid_size'], H['num_bin']
    prefix = str(join(save_dir, "{}_{}_{}".format(image_h, grid_h, num_bin)))
    if split=='test':
        print("### finished test split")
        return 0

    train_name = prefix + "_train.txt"
    val_name = prefix + "_val.txt"
    test_name = prefix + "_test.txt"
    dirname = prefix

    def valid_pickle_file(x):
        return x.endswith('.jpg') and isfile("{}/{}.pickle".format(dirname, x[:-4]))

    imgnames = [x for x in os.listdir(dirname) if valid_pickle_file(x)]
    train, val, test = [], [], []
    
    def read_list_from_file(f_):
        with open(f_, 'r') as h:
            filelist = h.read().splitlines()
            filelist = [x[:-4] for x in filelist]
        return filelist

    def write_list_to_file(f_, filelist):
        with open(f_, 'w') as h:
            for l in filelist:
                h.write("{}\n".format(l))

    trainImgs = read_list_from_file(ref.data_root / 'v1.1/train.txt')
    testImgs = read_list_from_file(ref.data_root / 'v1.1/test.txt')
    random.shuffle(trainImgs)
    
    trainImgs = trainImgs[200:]
    valImgs = trainImgs[:200]
    print("train: {} | val: {} | test: {}".format(len(trainImgs), len(valImgs), len(testImgs)))

    for f in imgnames:
        if '_' in f:
            prefix_ = f.split('_')[0]
            if prefix_ in trainImgs: train += [f]
            if prefix_ in valImgs: val += [f]
        else:
            prefix_ = f[:-4]
            if prefix_ in trainImgs: train += [f]
            if prefix_ in valImgs: val += [f]
            if prefix_ in testImgs: test += [f]
    
    print("train: {} | val: {} | test: {}".format(len(train), len(val), len(test)))
    write_list_to_file(train_name, train)
    write_list_to_file(val_name, val)


def create_dataset(H, split, use_mp = True):
    import time
    src_dir = ref.data_root / 'pointlines'
    save_dir = ref.junc_data_root / 'processed'
    with open(ref.data_root / 'v1.1/{}.txt'.format(split), 'r') as f:
        filelst = f.read().splitlines()
    
    filelst = [x[:-4] + '.pkl' for x in filelst]
    print(" #{split} images: {0}".format(len(filelst), split=split))
    start = time.time()
    load_data_and_save(H, src_dir=src_dir, save_dir=save_dir, filenames=filelst, use_mp = use_mp)
    print(" Elasped time: {:.2f}".format(time.time() - start))


