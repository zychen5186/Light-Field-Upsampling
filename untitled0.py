# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:51:21 2021

@author: brian
"""

import cv2
import numpy as np

img = cv2.imread("occluded_difmap.png")
mask = cv2.imread('occlusion_map.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
arr = np.array([1,4,5])
print(arr.any() == 5)


