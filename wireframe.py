import itertools
import json
import os
import random
import sys
from os.path import isdir, isfile, join, basename
from os import listdir
import time
from math import cos, pi, sin, tan

from joblib import Parallel, delayed
import pickle

import scipy.io as sio
from scipy.optimize import linear_sum_assignment
import numpy as np
from libs import *
from pathlib import Path
from tqdm import tqdm

divide_eps = 1e-20

img_dir = Path('data/v1.1/test/')
result_dir = Path('result')

def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def imwrite(imgname, img):
    # in case imgname is a Path obj.
    cv2.imwrite(str(imgname), img)

def imread(imgname, mode=1):
    return cv2.imread(str(imgname), mode)

class RayError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class CoordError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value


class Wireframe():
    def __init__(self, exp, imgname, junc_epoch=16, line_thresh=50, junc_thresh=0.1, use_mp=False, debug=False):
        self.use_mp = use_mp
        self.debug = debug

        self.line_threshold = line_thresh
        self.dataset_ = 'test'

        self.junc_epoch = str(junc_epoch)
        self.line_epoch = str(100)
        
        self.exp = exp
        self.junc_thresh = junc_thresh
        self.theta_thresh = 0.5

        self.junc_dir = result_dir / 'junc' / self.exp / self.junc_epoch
        self.line_dir = result_dir / 'linepx' / '0'
        self.wireframe_dir = result_dir / 'wireframe_{}_{}/{}'.format(self.junc_thresh, self.theta_thresh, self.line_threshold)
        #check_dir(self.wireframe_dir)

        self.line_map_size = 320
        self.img_dir = img_dir
        self.imgname = imgname

        # self.theta_dist_thresh = 1.6
        # self.theta_dist_relax = 4.5
        # self.dist_to_junction = 130
        self.theta_dist_thresh = 1.3
        self.theta_dist_relax = 4.0
        self.dist_to_junction = 130

        self.display = True if self.debug else False

    def load_img(self, in_=None):
        imgname = in_ if in_ is not None else self.imgname
        imgpath = self.img_dir / imgname
        img = imread(imgpath)
        self.h, self.w = img.shape[:2]
        self.max_side_length = max(self.h, self.w)
        self.img = img
        if self.debug:
            print(self.h, self.w, self.img.shape)

    def load_line(self, in_=None):
        imgname = in_ if in_ is not None else self.imgname

        line_name = self.line_dir  / "{}_line.npy".format(imgname[:-4])
        lineMap = np.load(line_name)
        if self.debug:
            print(lineMap.max())
        lineMap = cv2.resize(lineMap, (self.w, self.h), cv2.INTER_NEAREST)
        self.lineMap = cv2.threshold(lineMap, self.line_threshold, 255, cv2.THRESH_BINARY)[1]
        if self.debug:
            print(self.lineMap.max(), self.lineMap.shape)

    def load_junc(self, in_=None, theta_thresh=None, suffix='.pkl'):
        imgname = in_ if in_ is not None else self.imgname
        junc_thresh = int(self.junc_thresh * 10)
        junc_path = self.junc_dir / str(junc_thresh)
        pklfile = join(junc_path, imgname[:-4] + '_5.pkl')

        with open(pklfile, 'rb') as fn:
            try:
                d = pickle.load(fn)
            except TypeError:
                d = pickle.load(fn, encoding='latin1')

        h, w, thetas, junctions = d['h'], d['w'], d['thetas'], d['junctions']
        tmp_thetas = thetas

        njunctions, nthetas = [], []

        for junct, th  in zip(junctions, tmp_thetas):
            x, y = junct
            nx, ny = int(x), int(y)
            if nx < 0 or nx > w  or ny < 0 or ny > h:
                continue
                # raise CoordError("{}, {} exceed [0, {}], [0, {}]".format(nx, ny, w, h))
            
            if len(th) >= 2:
                njunctions.append((nx, ny))
                nthetas.append(th)

        self.pred_junctions = njunctions
        self.pred_thetas = nthetas
        if self.debug:
            print(len(njunctions), len(nthetas))

    def get_wireframe(self, gt=None):
        """
            Generating wireframes/lines from predicted junctions and line pixels.
            This func operates for a single image.
        """
        # theta_thresh = self.theta_thresh
        thetas = self.pred_thetas # [thresholding(ths, ths_confs, theta_thresh)[0] for ths, ths_confs in zip(self.pred_thetas, self.pred_confs)]

        junctions, nthetas = mergeDupJunctions(self.pred_junctions, thetas)
        #nthetas = mergeDupTheta(nthetas)
        
        img = self.img
        h, w = self.h, self.w
        junctions = np.array(junctions, dtype=np.float32)
        numJunc = junctions.shape[0]
        if self.display:
            showIm(name='linemap', img=self.lineMap)
            showJunctionPrediction(img, junctions, nthetas, imgname='junction map', display_=True)
        
        rays, rays_theta = possible_lines((h, w),  junctions, nthetas, img=img, DEBUG=False)

        # calculate d_min of junction and line_2
        ray_all = []
        ray_theta_all = []
        ray_map_junction = []
        junction_map_ray = [[] for _ in junctions]
        for idx, p in enumerate(junctions):
            rays_p = [(x, y, p[0], p[1]) for x, y, _, _ in rays[idx]]
            for l, t in zip(rays_p, rays_theta[idx]):
                ray_all.append(l)
                ray_theta_all.append(t)
                ray_map_junction.append(idx)
                junction_map_ray[idx].append(len(ray_all) - 1)
        ray_all = np.array(ray_all, dtype=np.float32)
        
        ## calculate several metrics:
        ##      1. the projection of junction on ray.
        ##      2. the distance of junction to ray.
        ##      3. the angle of angle with respect to the ends of ray.

        pointJuncOnRay, distJuncToRay, juncOnRay, juncRayTheta = calc_dist_theta(junctions, ray_all)

        numRay = ray_all.shape[0]
        T = np.zeros((numRay, numJunc))
        T -= 1
        within_T = np.zeros(T.shape).astype(np.float32)
        within_T -= 1.

        for i in range(numRay):
            point_idx = ray_map_junction[i]
            for j in range(numJunc):
                if point_idx == j:
                    continue
                on_line = juncOnRay[j, i, 2]
                delta_t = juncRayTheta[j, i]
                dist = distJuncToRay[j, i, 0]
                dist_to_junction = distJuncToRay[j, i, 3]

                if delta_t < self.theta_dist_thresh or (dist <= self.theta_dist_relax and dist_to_junction < self.dist_to_junction):
                    T[i, j] = delta_t
                    if on_line:
                        within_T[i, j] = delta_t

        mutual_T = -np.ones(T.shape)
        back_T = -np.ones(T.shape)

        for i in range(numRay):
            mapped_point_idx = ray_map_junction[i]
            indexes = np.nonzero(within_T[i,:] > -1 )[0]
            if len(indexes) == 0:
                continue

            for idx in indexes:
                if idx == mapped_point_idx:
                    continue
                subset_indexes = junction_map_ray[idx]
                #np.nonzero(lines_map_matrix == idx)[0]

                for si in subset_indexes:
                    if within_T[si, mapped_point_idx] > -1:
                        mutual_T[i, idx] = si
                        break
                    elif T[si, mapped_point_idx] > -1:
                        back_T[i, idx] = si

        ### line split 1
        mutual_T = mutual_T.astype(np.int32)

        im = img
        split_1 = []
        line2Line = np.zeros((numRay, numRay)).astype(np.int32)

        for i in range(numRay):
            k = ray_map_junction[i]
            point_indexes = np.nonzero(mutual_T[i, :]>-1)[0]

            x1, y1, x2, y2 = ray_all[i].tolist()

            if len(point_indexes) > 0:
                lengths = [(k, distJuncToRay[k, i, 3]) for k in point_indexes]
                lengths.sort(key=lambda x:x[1])
                endpoint_idx = lengths[0][0]

                line_idx = mutual_T[i, endpoint_idx]
                line2Line[i, line_idx] = 1
                line2Line[line_idx, i] = 1

        lineDistMatrix = pointDistMatrix(junctions[ray_map_junction, :], junctions[ray_map_junction, :])
        for i in range(numRay):
            nonzero_indexes = np.nonzero(line2Line[i, :] > 0)[0]
            if len(nonzero_indexes) > 1:
                dists = lineDistMatrix[i, nonzero_indexes]
                smallest_indexes = dists.argmin()
                for k, n_idx in enumerate(nonzero_indexes):
                    if k != smallest_indexes:
                        line2Line[i, n_idx] = 0
                        line2Line[n_idx, i] = 0

        for i in range(numRay):
            nonzero_indexes = np.nonzero(line2Line[i, :]>0)[0]
            assert len(nonzero_indexes) <= 1, len(nonzero_indexes)
            if len(nonzero_indexes) == 1:
                ni = nonzero_indexes[0]
                x1, y1, x2, y2 = (ray_all[ni, 2], ray_all[ni, 3], ray_all[i, 2], ray_all[i, 3])
                length_= sqrt((x1 - x2)**2 + (y1 - y2)**2)
                adding_line = False
                if length_ < self.max_side_length / 2.0:
                    adding_line = True
                else:
                    #coords = linespace((x1, y1), (x2, y2), img.shape[:2])
                    ratio, max_loc, _ = pixelRatio((x1, y1), (x2, y2), self.lineMap)
                    if ratio > 0.6:
                        adding_line = True

                if adding_line:
                    im = addLines(im, [(x1, y1, x2, y2)], display_=self.display, imgname="{} split 1".format(self.imgname), rand_color=False, color=(0, 255, 0), thickness=2)
                    split_1.append((ray_all[ni, 2], ray_all[ni, 3], ray_all[i, 2], ray_all[i, 3]))

        ### line split 2
        ### origin t22_4
        ## for all the cuts, only take the first
        split_2 = []
        for i in range(numRay):
            bi_point_indexes = np.nonzero(mutual_T[i, :] > -1)[0]
            online_point_indexes = np.nonzero(within_T[i, :] > -1)[0]

            bx, by = ray_all[i, 0], ray_all[i, 1]
            px, py = ray_all[i, 2], ray_all[i, 3]
            k = ray_map_junction[i]

            if distance((px, py), (bx, by)) < 0.05 * self.max_side_length:
                split_2.append((px, py, bx, by))
                im = addLines(im, [(px, py, bx, by)], display_=self.display, imgname="{} split 2".format(self.imgname), rand_color=False, color=(0, 0, 80), thickness=2)
                continue

            if len(bi_point_indexes) == 0 and len(online_point_indexes) == 0:
                ratio, max_loc, _ = pixelRatio((px, py), (bx, by), self.lineMap)
                if ratio < 0.05:
                    continue

                bx, by = max_loc

                intersects = intersections(np.array([px, py, bx, by], dtype=np.float32), np.array(split_1, dtype=np.float32))

                start_list =[(px, py)]
                end_list = []

                for int_ in intersects:
                    start_list.append(int_)
                    end_list.append(int_)
                end_list.append((bx, by))

                next_start = None
                previous_end = None

                first_flag = True
                for idx, (start, end) in enumerate(zip(start_list, end_list)):
                    if not first_flag:
                        break
                    if next_start is None:
                        next_start = start

                    line_ratio, max_pos_idx, max_pos_loc = pixelRatio(start, end, self.lineMap)
                    adding_line = False

                    valid_cond = line_ratio > 0.6 and max_pos_loc > 0.75
                    if idx == 0:
                        if distance(start, end) < self.max_side_length/20. or valid_cond:
                            adding_line = True
                    elif valid_cond:
                        adding_line = True

                    if adding_line:
                        previous_end = end
                        split_2.append((start[0], start[1], end[0], end[1]))
                        im = addLines(im, [(start[0], start[1], end[0], end[1])], display_=self.display, imgname="{} split 2".format(self.imgname), rand_color=False, color=(0, 0, 80), thickness=2)

                    else:
                        if idx == 0:
                            first_flag = False
                        next_start = None
                        previous_end = None

            elif len(bi_point_indexes) == 0 and len(online_point_indexes) > 0:
                online_point_indexes = online_point_indexes.tolist()
                online_point_indexes.sort(key=lambda x:distance(pointJuncOnRay[x, i], (px, py)))

                ratio, max_loc, _ = pixelRatio((px, py), (bx, by), self.lineMap)
                if ratio < 0.05:
                    continue
                bx, by = max_loc

                start_list =[(px, py)]
                end_list = []

                for on_idx in online_point_indexes:
                    start_list.append(pointJuncOnRay[on_idx, i])
                    end_list.append(pointJuncOnRay[on_idx, i])
                end_list.append((bx, by))

                next_start = None
                previous_end = None
                for idx, (start, end) in enumerate(zip(start_list, end_list)):
                    if next_start is None:
                        next_start = start

                    line_ratio, max_pos_idx, max_pos_loc = pixelRatio(start, end, self.lineMap)
                    adding_line = False

                    valid_cond = line_ratio > 0.6 and max_pos_loc > 0.75

                    if idx == 0:
                        if distance(start, end) < self.max_side_length/20. or valid_cond:
                            adding_line = True
                    elif valid_cond:
                        adding_line = True

                    if adding_line:
                        previous_end = end
                        split_2.append((start[0], start[1], end[0], end[1]))
                        im = addLines(im, [(start[0], start[1], end[0], end[1])], display_=self.display, imgname='{} split 2'.format(self.imgname), rand_color=False, color=(0, 0, 80), thickness=2)
                    else:
                        next_start = None
                        previous_end = None
        if len(split_1) == 0:
            print('{} no line from junction'.format(self.imgname))

        self.lines = split_1 + split_2


    def showLines(self):
        # save lines
        np_lines = np.array(self.lines)
        sio.savemat(self.wireframe_dir / (self.imgname[:-4] + '.mat'), {'lines':np_lines})

        # save visualization
        img = np.copy(self.img)
        im = addLines(img, self.lines, display_=False, color=(0, 255, 0), thickness=2)
        imwrite(self.wireframe_dir / self.imgname, im)


def calcAssignment(th1, th2, dist =7.5):
    H, W = len(th1), len(th2)
    costMatrix = np.zeros((H, W))
    m1 = np.array(th1, dtype=np.float32)
    m2 = np.array(th2, dtype=np.float32)
    m1 = np.reshape(m1, (H, 1))
    m2 = np.reshape(m2, (1, W))
    costMatrix = np.abs(m1- m2)
    costMatrix = np.minimum(costMatrix, 360 - costMatrix)
    costMatrix[costMatrix > dist] = 1000.
    ass_i, ass_j =  linear_sum_assignment(costMatrix)

    good, bad = [], []
    residual = 0.
    for i, j in zip(ass_i, ass_j):
        if costMatrix[i, j] <= dist:
            good.append((i, j))
            residual += costMatrix[i, j]
        elif costMatrix[i, j] == 1000.:
            bad.append((i, j))
    return good, bad, residual


def thresholding(ths, confs, thresh):
    nths, nconfs = [], []
    for t, c in zip(ths, confs):
        if c > thresh:
            nths.append(t)
            nconfs.append(c)
    if len(nths) == 0:
        return [], []
    zipped_list = list(zip(nths, nconfs))
    zipped_list.sort(key=lambda x: x[1])

    nths, nconfs = list(zip(*zipped_list))
    return nths, nconfs


######################################################
# Among all predicted junctions, merge the ones which
# are too close or have similar branches.
######################################################
def mergeDupJunctions(junctions, thetas):
    N = len(junctions)

    njunctions = []
    nthetas = []

    for i in range(N):
        if i == 0:
            njunctions.append(list(junctions[0]))
            nthetas.append(list(thetas[0]))
            continue

        dup_flag = False
        dup_idx = None
        match_list = []
        for j in range(len(njunctions)):
            dist_ij = distance(junctions[i], njunctions[j])
            if dist_ij <= 6:
                dup_flag = True
                dup_idx = j
                good, bad, _ = calcAssignment(thetas[i], nthetas[j])
                if len(good) >= 1:
                    match_list.append((j, good))
                else:
                    match_list =[(j, [])]
                break

            elif dist_ij <= 10:
                good, bad, _ = calcAssignment(thetas[i], nthetas[j])
                if len(good) >= 1:
                    match_list.append((j, good))
            else:
                continue

        if match_list:
            match_list.sort(key=lambda x:len(x[1]))

        if dup_flag or len(match_list) > 0:
            matched_idx = []

            if not dup_flag:
                dup_idx, matched_idx = match_list[-1]
            else:
                dup_idx, matched_idx = match_list[0]

            new_thetas = []
            x1, y1 = junctions[i]
            x2, y2 = njunctions[dup_idx]
            x, y = (x1 + x2)/2., (y1 + y2)/2.
            njunctions[dup_idx] = (x, y)

            # merge junctions[i] with njucntions[idx], if a branch is matched, then not add to the new junction.
            dup_indexes_theta = [k1 for k1, _ in matched_idx]
            dup_indexes_ntheta = [k2 for _, k2 in matched_idx]

            for t1, t2 in matched_idx:
                new_thetas.append( (thetas[i][t1] + nthetas[dup_idx][t2])/2. )

            for idx_1, t in enumerate(thetas[i]):
                if idx_1 not in dup_indexes_theta:
                    new_thetas.append(t)

            for idx_2, t in enumerate(nthetas[dup_idx]):
                if idx_2 not in dup_indexes_ntheta:
                    new_thetas.append(t)

            nthetas[dup_idx] = new_thetas
        else:
            njunctions.append(junctions[i])
            nthetas.append(list(thetas[i]))

    return njunctions, nthetas

def mergeDupTheta(thetas, theta_thresh=4.):
    new_thetas = [[] for _ in thetas]
    for idx, ths in enumerate(thetas):
        num = len(ths)
        new_ths = []
        ths.sort()
        for i, t in enumerate(ths):
            dup_flag = False
            dup_idx = None
            for j, new_t in enumerate(new_ths):
                if theta_dist(t, new_t) < theta_thresh:
                    dup_flag = True
                    dup_idx = j
                    break
            if not dup_flag:
                new_ths.append(ths[i])
            else:
                new_ths[dup_idx] = 0.5 * new_ths[dup_idx] + 0.5 * t
        new_thetas[idx] = new_ths
    return new_thetas


def theta_dist(th1, th2):
    d1 = min(th1, th2)
    d2 = max(th1, th2)
    return min(d2 - d1, d1 + 360 - d2)

def innerProduct(A, B, axis=-1):
    return np.sum(np.multiply(A, B),  axis=axis)

def calc_dist_theta(points, lines, geometric_ = False):
    lines = lines.astype(np.float32)

    p1 = lines[np.newaxis, :, :2]
    p2 = lines[np.newaxis, :,  2:4]
    p = points[:, np.newaxis, :].astype(np.float32)

    # the intersection is px, py
    p1p = p - p1
    p2p = p - p2
    p1p2 = p2 - p1

    p1pm = innerProduct(p1p, p1p2) / (np.sum(np.square(p1p2), axis=-1) + eps)
    p1pm = np.expand_dims(p1pm, axis=-1) * p1p2
    pm = p1 + p1pm
    p2pm = -p1p2 + p1pm

    p1p2_l = np.sqrt(np.sum(np.square(p1p2), axis=-1))

    p1pm_s = innerProduct(p1pm, p1p2)/(p1p2_l + eps)
    p2pm_s = innerProduct(p2pm, -p1p2)/(p1p2_l + eps)

    ppm_l = np.linalg.norm(p - pm, axis=-1)

    on_line_1 = p1pm_s >= 0
    on_line_2 = p2pm_s >= 0
    on_line = np.logical_and(on_line_1, on_line_2)


    pp1_l = np.linalg.norm(p - p1, axis=-1)
    pp2_l = np.linalg.norm(p - p2, axis=-1)

    dist_endpoint = np.minimum(pp1_l, pp2_l)
    short_dist = ppm_l if geometric_ else np.where(on_line, ppm_l, dist_endpoint)
    dist = np.stack([short_dist, ppm_l, p1pm_s, p2pm_s], axis=-1)

    theta_p1 = innerProduct(p1p, p1p2)/(p1p2_l * pp1_l + eps)
    theta_p2 = innerProduct(p2p, -p1p2)/(p1p2_l * pp2_l + eps)

    theta_p1 = np.arccos(np.clip(theta_p1, -1., 1.)) * 180. / pi
    theta_p2 = np.arccos(np.clip(theta_p2, -1., 1.)) * 180. / pi

    theta_p1 = np.minimum(theta_p1, 180. - theta_p1)
    theta_p2 = np.minimum(theta_p2, 180. - theta_p2)

    theta_p = np.minimum(theta_p1, theta_p2)
    on_line = np.stack([on_line_1, on_line_2, on_line], axis=-1)

    return pm, dist, on_line, theta_p


def possible_lines(img_size, points, thetas, img=None, DEBUG=False):
    (h, w) = img_size
    assert len(points) == len(thetas), "number of points is not euqal to len(thetas) {} vs {}".format(len(points), len(thetas))
    possible_lines = [[] for _ in points]
    possible_lines_theta = [[] for _ in points]

    for idx, (x, y) in enumerate(points):
        ths = thetas[idx]
        line_to_show = []
        for t in ths:
            if 89.8 < t < 90.2:
                line_se = (x, h, x, 0)
            elif 269.8 <t < 270.2:
                line_se = (x, 0, x, h)
            elif 0 <= t <= 0.2 or  359.8<= t<=360:
                line_se = (w, y, 0, y)
            elif 179.8 <= t <= 180.2:
                line_se = (0, y, w, y)
            else:
                tr = t * pi / 180.
                # line: y = (x - x0) * tan(\theta) + y0

                # 1. x = 0., w, y should belong to
                y_0 = - x * tan(tr) + y
                y_w = (w - x) * tan(tr) + y

                x_0 = (0 - y) / tan(tr) + x
                x_h = (h - y) / tan(tr) + x

                points_to_select = []
                if 0 <= y_0 <= h:
                    points_to_select.append((0, y_0))
                if 0 <= y_w <= h:
                    points_to_select.append((w, y_w))

                if 0 <= x_0 <= w:
                    points_to_select.append((x_0, 0))
                if 0 <= x_h <= w:
                    points_to_select.append((x_h, h))

                assert len(points_to_select) == 2, "only 2 points can be valid."
                p1, p2 = points_to_select
                c, s = cos(tr), sin(tr)
                ip1 = (p1[0] - x) * c + (p1[1] - y) * s
                ip2 = (p2[0] - x) * c + (p2[1] - y) * s

                if ip1 > 0:
                    line_se = (p1[0], p1[1], p2[0], p2[1])
                elif ip2 > 0:
                    line_se = (p2[0], p2[1], p1[0], p1[1])
                elif ip1 < 0:
                    line_se = (p2[0], p2[1], p1[0], p1[1])
                elif ip2 < 0:
                    line_se = (p1[0], p1[1], p2[0], p2[1])
                else:
                    raise RayError('no intersection found of ray')

            possible_lines[idx].append(line_se)
            possible_lines_theta[idx].append(t * pi/ 180.)
            line_to_show.append((x, y, line_se[0], line_se[1]))
        # im = addLines(im, line_to_show, display_ = True if DEBUG else False, color=(0, 255, 0), thickness=2)

    return possible_lines, possible_lines_theta

def deltaTheta(ep, sp, pt):
    if not isinstance(pt, np.ndarray):
        pt = np.array(pt, dtype=np.float32)
    dir_v = np.stack([np.cos(pt), np.sin(pt)], axis=-1)

    ep = ep[:, np.newaxis,  :]
    sp = sp[np.newaxis, :,  :]
    v = ep - sp

    dir_v = dir_v[np.newaxis, :, :]

    se = np.linalg.norm(v, axis=-1, keepdims=False)
    cosThetas = innerProduct(v, dir_v) / (se + eps)
    cosThetas = np.clip(cosThetas, -1, 1)
    angles = np.arccos(cosThetas)
    angles = angles * 180./ pi
    return angles


def pointDistMatrix(pts1, pts2):
    v = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    dist = np.linalg.norm(v, axis=2)

    return dist

def intersections(line, lines_all):
    p = line[:2]

    intersects = []
    for la in lines_all:
        intersect = intersectionOfTwoLines(line, la)
        if intersect is None :
            continue
        ip1 = np.inner(line[:2] - intersect, line[2:] - intersect)
        ip2 = np.inner(la[:2] - intersect, la[2:] - intersect)
        if ip1 >= 0 or ip2 >= 0:
            continue
        intersects.append(intersect)
    intersects.sort(key=lambda x:((x[0] - p[0])**2 + (x[1] - p[1])**2))
    nodup = []
    for i, it_ in enumerate(intersects):
        dup_flag = False
        dup_idx = None
        for j, nd_ in enumerate(nodup):
            if distance(it_, nd_) < 3.:
                dup_flag = True
                dup_idx = j
                break
        if not dup_flag:
            nodup.append(it_)

    return nodup


def pixelRatio(p1, p2, M):
    coords = linespace(p1, p2, M.shape)
    map_value = M[coords]
    #print("\n map_value: ", map_value, "\n"

    ratio, max_idx_loc, max_idx = ratioSeq(map_value)
    max_idx = int(max_idx)
    if len(coords[0]) == 0:
        return 0, p1

    locx = coords[1][max_idx]
    locy = coords[0][max_idx]

    return ratio, (locx, locy), max_idx_loc


def ratioSeq(seq):
    num = len(seq)
    nz_indexes = np.nonzero(seq)[0]
    if len(nz_indexes) == 0:
        return 0., 0., 0.
    max_idx = nz_indexes.max()
    return float(len(nz_indexes)) / float(max_idx + 1), float(max_idx + 1)/ float(num), max_idx

def linespace(p1, p2, shape):
    x1, y1 = p1
    x2, y2 = p2
    h, w = shape

    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)

    x1 = min(max(x1, 0), w - 1)
    x2 = min(max(x2, 0), w - 1)
    y1 = min(max(y1, 0), h - 1)
    y2 = min(max(y2, 0), h - 1)

    num_x = max(x1, x2) - min(x1, x2) + 1
    num_y = max(y1, y2) - min(y1, y2) + 1

    if num_x < num_y:
        xlist = np.linspace(x1, x2, num=num_y)
        ylist = np.linspace(y1, y2, num=num_y)
    else:
        xlist = np.linspace(x1, x2, num=num_x)
        ylist = np.linspace(y1, y2, num=num_x)

    xlist = xlist.astype(np.int32)
    ylist = ylist.astype(np.int32)

    ylist[ylist > (h -1)] = h -1
    xlist[xlist > (w - 1)] = w - 1
    coords = np.vstack((ylist, xlist))
    return tuple(coords)


def angleBetweenLines(l1, l2):
    x = l1[:2] - l1[2:4]
    y = l2[:2] - l2[2:4]
    theta = np.arccos( np.clip(innerProduct(x, y)/(np.linalg.norm(x) * np.linalg.norm(y) + eps), -1., 1.)) * 180./ pi
    return np.minimum(theta, 180. - theta)


def angleBetweenLinesMatrix(L1, L2):
    M, N = L1.shape[0], L2.shape[0]
    x = L1[:, :2] - L1[:, 2:4]
    y = L2[:, :2] - L2[:, 2:4]
    x = x[:, np.newaxis, :]
    y = y[np.newaxis, :, :]

    theta = np.arccos( np.clip(innerProduct(x, y)/(np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + eps), -1., 1.)) * 180./ pi
    return np.minimum(theta, 180. - theta)

def process_wireframe(exp_name, in_, junc_thresh, line_thresh, debug=False):
    wf = Wireframe(exp_name, in_, junc_thresh=junc_thresh, line_thresh=line_thresh, debug=debug)
    wf.load_img()
    wf.load_line()
    wf.load_junc(theta_thresh=None)
    wf.get_wireframe()
    wf.showLines()

def main():
    with open(img_dir / '..' / 'test.txt', 'r') as fn:
        imgnames = fn.read().splitlines()
    print("# test images: {}".format(imgnames.__len__()))
    
    theta_thresh = 0.5
    junc_thresh = 0.5
    exp_name = '1'
    debug = False
    use_mp = True if not debug else False
    njobs = 30
    
    print("junction threshold, theta threshold: {}, {}".format(junc_thresh, theta_thresh))

    line_thresholds = [100] if debug else [2, 6, 10, 20, 30, 50, 80, 100, 150, 200, 250, 255]

    
    for line_thresh in line_thresholds:
        wireframe_dir = result_dir / 'wireframe_{}_{}/{}'.format(junc_thresh, theta_thresh, line_thresh)
        check_dir(wireframe_dir)
        print("== line threshold: {}".format(line_thresh))
        if use_mp:
            Parallel(n_jobs=njobs)(delayed(process_wireframe)(exp_name, in_, junc_thresh, line_thresh)
                for in_ in tqdm(imgnames))  # max_len default to 1.
        else:
            for in_ in tqdm(imgnames):
                process_wireframe(in_, junc_thresh, line_thresh, debug=debug)

if __name__ == '__main__':
    main()
