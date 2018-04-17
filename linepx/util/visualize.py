import os
import math
import numpy as np
import cv2

def renderImgTable(inputImgs, ncoln, imgDir):
    htmlStr = '<html><head><style>* {font-size: 24px;}</style></head><body><table>\n'
    for row in range(math.ceil(len(inputImgs) / ncoln)):
        htmlStr += '<tr><td>%d</td>' % row
        for coln in range(ncoln):
            ID = row * ncoln + coln
            cv2.imwrite(os.path.join(imgDir, '%04d.png' % ID), inputImgs[ID])
            htmlStr += '<td><img width=240 src="im/%04d.png"></img></td>' % ID
        htmlStr += '</tr>\n'
    htmlStr += '</table></body></html>\n'
    return htmlStr

def writeImgHTML(inputImgs, epoch, split, ncoln, opt):
    htmlFile = 'index.html'
    rootDir = os.path.join(opt.resume, str(epoch) + '_' + split)
    if not os.path.exists(rootDir):
        os.makedirs(rootDir)

    imgDir = os.path.join(rootDir, 'im')
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)

    with open(os.path.join(rootDir, htmlFile), 'w') as f:
        f.write('<html>\n<body>\n')
        table = renderImgTable(inputImgs, ncoln, imgDir)
        f.write(table)
        f.write('</body>\n</html>\n')
