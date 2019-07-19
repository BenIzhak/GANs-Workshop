# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:25:12 2019

@author: Rotem
"""

import cv2

def resize (numOfImages):
    for i in range(0, numOfImages):
    path = 'path\\to\\results\\result-' + str(i) + '.png'
    img = cv2.imread(path)

    dim = (64, 64)
    img = cv2.resize(img, dim) 
    
    
    cv2.imwrite('path\\to\\final\\results\\result-' + str(i) + '.png', img)

