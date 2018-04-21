import numpy as np
all_colors = np.random.random_integers(0, high=255, size=(20, 3)) ## for visualizing the lines

import cv2

from math import cos, sin, pi, sqrt
from scipy.optimize import linear_sum_assignment

import dominate
from dominate.tags import *
import os
from os.path import isdir

eps = 1e-25
same_point_eps = 3.

def addLines(img, lines, imgname='adding_lines', color=(0,0,255), rand_color=False, thickness=1, display_=False):
    newim = np.copy(img)
    if lines is None or len(lines) == 0:
        return newim

    for x1, y1, x2, y2 in lines:
        l_color = tuple(np.random.random_integers(0, high=255, size=(3))) if rand_color else color
        cv2.line(newim, (int(x1), int(y1)), (int(x2), int(y2)), l_color, thickness)
    if display_:
        showIm(imgname, newim)
    return newim

def addPoints(img, points, imgname = 'adding_points', color = (0, 255, 255), thickness = 2, display_ = False):
    newim = np.copy(img)
    for x,y in points:
        cv2.ellipse(newim, (int(x), int(y)), (thickness, thickness),
                0, 0, 360, color, -1)
    if display_:
        showIm(imgname,newim)
    return newim


def showJunctionPrediction(img, junctions, thetas,
                           color = (0, 255, 0), 
                           imgname = 'junction', 
                           display_=False, 
                           thickness=2,
                           rand_color = False
                           ):
    newim = np.copy(img)


    for (x, y), theta in zip(junctions, thetas):
        assert len(theta) >= 0
        if len(theta) < 2:
            continue
        l_color = tuple(all_colors[len(theta), :]) if rand_color else color
        cv2.ellipse(newim, (int(x), int(y)), (2,2), 0, 0, 360, l_color, -1)
        for t in theta:
            if isinstance(t, tuple):
                z = t[0]
            else:
                z = t
            x1, y1 = (x + 12.5 * cos(z * pi / 180.),
                  y + 12.5 * sin(z * pi / 180.))
            cv2.line(newim, (int(x), int(y)),
                 (int(x1), int(y1)), l_color, thickness)
            cv2.line(newim, (int(x), int(y)),
                 (int(x1), int(y1)), l_color, thickness)
    if display_:
        showIm(imgname,newim)
    
    return newim


def intersectionOfLines(A, B):
        """
            A and B are M and N points. (Mx2 and Nx2)
        """
        #print A.shape, B.shape
        da = A[:, 2:4] - A[:, :2]
        db = B[:, 2:4] - B[:, :2]
        dp = A[:, np.newaxis, :2] - B[np.newaxis, :, :2]
        dap = np.hstack((-da[:, 1], da[:, 0]))

        print(da.shape, db.shape, dp.shape, dap.shape)


        if dap.ndim == 1:
            dap = dap[np.newaxis, :]

        denom = np.sum(np.multiply(dap[:, np.newaxis, :], db[np.newaxis, :, :]), axis=2)
        num = np.sum(np.multiply(dap[:, np.newaxis, :], dp), axis=2)
        
        print(denom.shape, num.shape)

        #valid = denom !=0
     
        tmp =  num / (denom.astype(float) + eps)
        intersect = tmp[:, :, np.newaxis]* db[np.newaxis, :, :] + B[np.newaxis, :, :2]
        
        print(intersect.shape)

        return intersect


def intersectionOfTwoLines(A, B):
    """
        A and B are M and N points. (Mx2 and Nx2)
    """
    da = A[2:4] - A[:2]
    db = B[2:4] - B[:2]
    dp = A[:2] - B[:2]

    dap = np.hstack((-da[1], da[0]))
    #print da.shape, db.shape, dp.shape, dap.shape
    denom = np.sum(np.multiply(dap, db))
    num = np.sum(np.multiply(dap, dp))
    #print denom.shape, num.shape
    if denom == 0:
        return None 
    tmp =  num / denom.astype(float) 
    intersect = tmp * db + B[:2]
    return intersect


def angleOfLine(p1, p2):
    """
    p1 -> p2
    """
    x1, y1 = p1
    x2, y2 = p2

    dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)

    if abs(y1 - y2) <= 0.1:
        if x1 > x2:
            return 180.
        else:
            return 0.
    elif y1 < y2:
        theta = acos((x2 - x1) / dist)
        return theta / pi * 180.
    else:
        theta = acos((x2 - x1) / dist)
        return 360. - theta / pi * 180.

def lineScale(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1- x2)**2 + (y1 - y2)**2)

def SamePoint(p1, p2, eps = None):
        if eps is None:
            eps = same_point_eps
        x1, y1 = p1
        x2, y2 = p2
        dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist < eps

def EqualPoint(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return x1 == x2 and y1 == y2


def distance(p1, p2):
    return sqrt( (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2 )


def showIm(name='test', img = None):
    if img is None:
        return
    else:
        #print "show img"
        #cv2.resizeWindow(name, 1000, 1000)
        if img.max() > 10:
            img = img.astype(np.uint8)
        cv2.imshow(name, img)
        key = cv2.waitKey(0)
        if key == 'a':
            cv2.destroyAllWindows()
        if key == 27:
            return None

def calcAssignment(th1, th2, dist = 10.):
    H = len(th1)
    W = len(th2)
    costMatrix = np.zeros((H, W))
    m1 = np.array(th1, dtype=np.float32)
    m2 = np.array(th2, dtype=np.float32)
    m1 = np.reshape(m1, (H, 1))
    m2 = np.reshape(m2, (1, W))
    costMatrix = np.abs(m1- m2)
    costMatrix = np.minimum(costMatrix, 360 - costMatrix)
    costMatrix[costMatrix > dist] = 1000.
    ass_i, ass_j =  linear_sum_assignment(costMatrix)
    
    good = []
    bad = []
    residual = 0.
    for i, j in zip(ass_i, ass_j):
        if costMatrix[i, j] <= dist:
            good.append((i, j))
            residual += costMatrix[i, j]
        elif costMatrix[i, j] == 1000.:
            bad.append((i, j))

def minDist(p, pts):
    dists = [distance(p, pt) for pt in pts]
    
    min_dist = min(dists)
    min_idx = dists.index(min_dist)
    return min_dist, min_idx

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

def removeDupJunctions(junctions, thetas):
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

def removeDupTheta(thetas, theta_thresh=4.):
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


def innerProduct(A, B, axis=-1):
    return np.sum( np.multiply(A, B),  axis=-1)


def calc_dist_theta(points, lines, geometric_ = False):
    lines = lines.astype(np.float32)

    p1 = lines[np.newaxis, :, :2]
    p2 = lines[np.newaxis, :,  2:4]
    p = points[:, np.newaxis, :].astype(np.float32)
    
    # the intersection is px, py
    p1p = p - p1
    p2p = p - p2
    p1p2 = p2 - p1
    #print p1p.shape, p1p2.shape

    p1pm = innerProduct(p1p, p1p2) / (np.sum(np.square(p1p2), axis=-1) + eps) 
    p1pm = np.expand_dims(p1pm, axis=-1)* p1p2
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


def generate_html(imglst, folder, table_keys=['gt', 'lsd'],  img_name=True):
    possible_keys = {'gt' : "groudtruth junction", 
                     "lsd": "LSD, IPOL 2012, TPAMI 2010."
                     }

    h = html()
    keys = table_keys
    values = [possible_keys[k] if k in possible_keys.keys() else k for k in keys]

    with h.add(body()).add(div(id='content')):
        h1('View of Results.')
        with table().add(tbody()):
            l = tr()
            if img_name:
                l += th('Imgname')
            for k in values:
                l += th(k)

            for in_ in imglst:
                l = tr()
                if img_name:
                    l += td(in_)
                for k in keys:
                    l += td(img(width =250, src='%s_%s.png'%(in_, k)) )
    folder_name = folder
    if not isdir(folder_name):
        os.makedirs(folder_name)

    with open('%s/index.html'%(folder_name), 'w') as fn:
        print >> fn, h

