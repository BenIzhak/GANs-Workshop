# -*- coding: utf-8 -*-
import cv2
   
def bitwiseAnd(numOfImages):
    for i in range(1, numOfImages):
        path1 = 'path\\to\\HED\\hed-' + str(i) + '.png'
        img1 = cv2.imread(path1)
        path2 = 'path\\to\\Canny\\edges\\canny-' + str(i) + '.png'
        img2 = cv2.imread(path2)
        
        img_bwa = cv2.bitwise_and(img1,img2)
        
        cv2.imwrite('path\\to\\final\\edges\\img-' + str(i) + '.png', img_bwa)
