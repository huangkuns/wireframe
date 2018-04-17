import os
import cv2
import numpy as np
import scipy.io as sio
import pickle
import ref
from tqdm import tqdm

def intx(x):
    return(int(x[0]), int(x[1]))

def processData(split):
    # Where to load raw images
    dataPath = ref.data_root / 'v1.1' / split
    imgnames = [x for x in os.listdir(dataPath) if x.endswith('.jpg')]
    # Where to load annotations
    annPath = ref.data_root / 'pointlines'
    # Where to store processed images
    outPath = ref.data_root / 'linepx' / 'processed'

    inpSize = ref.input_size

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for idx, in_ in enumerate(tqdm(imgnames)):
        # print("img {}".format(idx))
        img_path = str(dataPath / in_)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        line = np.zeros((inpSize, inpSize))

        with open(annPath / "{}{}".format(in_[:-4], ref.ext), 'rb') as f:
            target = pickle.load(f, encoding='latin1')
            points = target['points']
            lines = target['lines']
            lines_original = []
            for i, j in lines:
                imgSize = np.array((w, h))
                start = np.array( points[i] ) * inpSize / imgSize
                end = np.array( points[j] ) * inpSize / imgSize
                lines_original.append(points[i] + points[j])
                dist = np.linalg.norm(end - start) / (inpSize * np.sqrt(2))
                line = cv2.line(line, intx(start), intx(end), 255 * dist, 2)
            lines_original = np.array(lines_original)

        save_imgname = outPath / "{}{}".format(in_[:-4], '_rgb.png')
        lineName = outPath / "{}{}".format(in_[:-4],'_line.png')
        lineOrigName = outPath / "{}{}".format(in_[:-4],'_line.mat')
        outImg = cv2.resize(img, (inpSize, inpSize))

        sio.savemat(lineOrigName, {'lines':lines_original})
        cv2.imwrite(str(save_imgname), outImg)
        cv2.imwrite(str(lineName), line)
