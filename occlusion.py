# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:16:35 2021

@author: brian
"""
import cv2
import numpy as np

right = cv2.imread("warping_dataset/bike/bike_dis/9_right.png-d.png")
left = cv2.imread("warping_dataset/bike/bike_dis/7_right.png-d.png")
right = right[:,:,0]
left = left[:,:,0]
height = right.shape[0]
width = right.shape[1]
occlusion_map = np.zeros((height,width))

right = ((right/255) * (48.604034-0.000357) + 0.000357)
left = ((left/255) * (48.905609-0.000246) + 0.000246)

right = right.astype(np.uint8)
left = left.astype(np.uint8)

for i in range(height):
    for j in range(width):
        if((j-left[i,j])>=0):
            if(abs(left[i,j] - right[i,j-left[i,j]]) > 10):
                occlusion_map[i,j] = 255

# =============================================================================
# occlusion_map2 = np.zeros((height,width))
# for i in range(height):
#     for j in range(width):
#         if((j + right[i,j]) < width):
#             if(abs(right[i,j] - left[i,j+right[i,j]]) > 10):
#                 occlusion_map2[i,j] = 255
#                 
# 
# diff = np.zeros((height,width))
# for i in range(height):
#     for j in range(width):
#         if(occlusion_map[i,j]==occlusion_map2[i,j] and occlusion_map2[i,j]==255):
#             diff[i,j] = 255
#             
# =============================================================================
cv2.imshow("left_", occlusion_map)      
# =============================================================================
# cv2.imshow("right", occlusion_map2)#右視圖會找到的occlusion
# cv2.imshow("dif",diff)
# =============================================================================
