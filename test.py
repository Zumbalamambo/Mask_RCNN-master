# -*- coding: utf-8 -*-
"""
Created on 2018/7/25 12:03
@author: gzp
Func: 
"""
import numpy as np
import cv2
img = np.zeros([300, 300, 3], np.uint8)
left_bottom_coord = (100, 100)
cv2.putText(img, 'there 0 error(s):', left_bottom_coord, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,255),thickness=1)
cv2.imshow('', img)
cv2.waitKey(0)