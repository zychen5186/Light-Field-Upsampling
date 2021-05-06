# create a light field data set without hardware support

import numpy as np
import cv2

for i in range(81):
    num = str(i).zfill(3)
    ##### dataset這裡改
    img = cv2.imread("dataset/HCI dataset/boardgames/input_Cam" + num + ".png") #original data set
    #####
    if(i==40):
        cv2.imwrite("test_data/input_"+num+".png",img)
        continue
    scale_percent = 1/np.sqrt(8) # percent of original size
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test_data/input_"+num+".png",img)