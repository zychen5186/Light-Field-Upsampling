import cv2
import numpy as np

def upsample_data(directory,h,w):
    for i in range(81):
        num = str(i).zfill(3)
        ##### dataset這裡改
        img = cv2.imread(directory + "input_" + num + ".png") #original data set
        #####
        if(i==40):
            scale_percent = 1/np.sqrt(8) # percent of original size
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        img = cv2.resize(img, (h, w), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("upsampled_data/input_"+num+".png",img)