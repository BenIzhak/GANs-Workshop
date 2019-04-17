import cv2
import numpy as np
from matplotlib import pyplot as plt


def getEdgeMap(path, lwT, upT):
    img = cv2.imread(path, 0)
    edges = cv2.Canny(img, lwT, upT)
    return edges

def convertToBandW(img, i):
    im_bw = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('edges\\' + 'e-' + str(i) + '.png', im_bw)
    

def convertImg(startIndex, endIndex, lwT = 120, upT = 300):
    for i in range(startIndex, endIndex + 1):
        path = 'frog-' + str(i) + '.png'
        edges = getEdgeMap(path, lwT, upT)
        convertToBandW(edges, i)
        
        
convertImg(1, 7796, lwT = 120, upT = 300)