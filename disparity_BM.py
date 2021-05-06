# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:19:54 2020

@author: brian
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread("left.png",0)
imgR = cv2.imread("right.png",0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite("dispatrity_from_disparity_map_1.png", disparity)
plt.imshow(disparity,'gray')
plt.show()