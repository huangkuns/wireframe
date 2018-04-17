import os
import cv2
from math import pi, sin, cos
try:
    import tensorflow as tf
except:
    LOG = False
from io import BytesIO
import numpy as np
import itertools

import ref
from model.networks.generate_cell_anchors import generate_anchors, generate_bin_anchors
from datasets.utils import resize_theta


class Plotter(object):
    def __init__(self, H):
        self.init_plot_param(H)

    def init_plot_param(self, H):
        self.junction_color = (0, 0, 255)
        self.line_color = (0, 255, 0)
        self.line_length = 12
        self.num_bin = H['num_bin']
        self.grid_size = H['grid_size']
        self.grid_num = self.grid_size**2
        self.stride = H['region_size']
        
        bin_width = 360. / float(self.num_bin)
        self.centers = np.array(
            [i * bin_width for i in range(self.num_bin)], dtype=np.float32)
        self.junc_anchors = generate_anchors((self.grid_size, self.grid_size), stride=self.stride).reshape((-1, 2))
        self.bin_anchors = generate_bin_anchors(self.num_bin)
        self.bin_width = 360./float(self.num_bin)
        
        self.pixel_mean = ref.pixel_mean
        
    @staticmethod
    def convert_tensor_to_numpy(tp):
        ntp = [x.numpy() for x in tp]
        return ntp

    def revert_junc_to_absoluteValue(self, junc_conf, junc_res):
        junc_res = junc_res.transpose((1, 2, 0)).reshape((self.grid_num, 2))
        junc_conf = junc_conf.reshape((self.grid_num))
        junc_pos = junc_res + self.junc_anchors

        return junc_conf, junc_pos
    
    def revert_bin_to_absoluteValue(self, bin_conf, bin_res):
        bin_res = bin_res.transpose((1, 2, 0)).reshape((self.grid_num, self.num_bin))
        bin_conf = bin_conf.transpose((1, 2, 0)).reshape((self.grid_num, self.num_bin))
        angles = bin_res + self.bin_anchors
        return bin_conf, angles
    
    def plot_junction_simple(self, image, output_result, junc_thresh=0.5, theta_thresh=0.5, plot_=True, size_info=None, keep_conf=False):
        """ adding junctions to images.
            tag: 'pred' or 'gt'
            for "pred", the junction is absoluate position.
            for "gt", it's relative position, add it here.

        """
        only_junction = True
        if len(output_result) == 4:
            only_junction = False
            junc_conf, junc_res, bin_conf, bin_res = output_result
        elif len(output_result) == 2:
            only_jucntion = True
            junc_conf, junc_res = output_result
        else:
            raise
            
        image = image.transpose((1, 2, 0))
        image = image + self.pixel_mean
        
        nimage = image.copy()
        h, w = nimage.shape[:2]
    
        grid_h = grid_w = self.grid_size
        self.grid_num = grid_h * grid_w
        
        cell_size = self.stride
        
        junc_conf, junc_pos = self.revert_junc_to_absoluteValue(junc_conf, junc_res)
        bin_conf, angles = self.revert_bin_to_absoluteValue(bin_conf, bin_res)

        valid_junc_idx = np.nonzero(junc_conf > junc_thresh)[0]
        junctions = junc_pos[valid_junc_idx]
        thetas = angles[valid_junc_idx, :]
        theta_confs = bin_conf[valid_junc_idx, :]
        
        if keep_conf:
            thresh_bin = lambda x:(x[0][np.nonzero(x[1]> theta_thresh)], x[1][np.nonzero(x[1] > theta_thresh)])
            tmp = [thresh_bin(x) for x in zip(thetas, theta_confs)]
            nthetas = [x.tolist() for x, y in tmp]
            nconfs = [y.tolist() for x, y in tmp]
        else:
            thresh_bin = lambda x:x[0][np.nonzero(x[1] > theta_thresh)]       
            nthetas = [thresh_bin(x).tolist() for x in zip(thetas, theta_confs)]
            nconfs = []

        if size_info is not None:
            origin_h, origin_w = size_info
            sw = float(origin_w)/float(w)
            sh = float(origin_h)/float(h)
            junctions = [(sw * x, sh * y) for x, y in junctions]
            nthetas = resize_theta(nthetas, [sw, sh])        
            nimage = cv2.resize(nimage, (origin_w, origin_h))
            w, h = origin_w, origin_h
        if not plot_:
            return nimage, junctions, nthetas, nconfs

        for idx, (jx, jy) in enumerate(junctions):
            if not ((jx >= 0 and jx <= w) and (jy >= 0 and jy <= h)):
                continue
            try:
                cv2.circle(nimage, (int(jx), int(jy)), 2,  self.junction_color, -1)
            except OverflowError:
                print(jx, jy)
                raise
            if only_junction:
                continue

            for th in nthetas[idx]:
                rad = th * pi / 180.0
                dx, dy = (cos(rad), sin(rad))
                xe, ye = (jx + self.line_length * dx,
                          jy + self.line_length * dy)
                xe, ye = min(max(xe, 0), w), min(max(ye, 0), h)
                cv2.line(nimage, (int(jx), int(jy)), (int(xe), int(ye)), self.line_color, 2)
        return nimage, junctions, nthetas, nconfs
