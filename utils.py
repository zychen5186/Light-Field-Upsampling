# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:14:13 2021

@author: brian
"""
import numpy as np
from openpyxl import Workbook
import cv2

def recoverDisValue(height, width):
    print("Restoring true disparity map...")
    disp_arr_sub = np.zeros((21, height, width))
    disp_arr_mid = disp_arr_sub.copy()  # 以中間為底跟周圍16張的disparity map
    f = open("warping_dataset/bike/bike_dis/disparity_record.txt", mode='r')
    for i in range(17):
        if(int(f.readline().rstrip('\n')) == i):
            temp_disp = cv2.imread(
                "warping_dataset/bike/bike_dis/" + str(i) + "_left.png-d.png")
            min_disp = float(f.readline().rstrip('\n'))
            max_disp = float(f.readline().rstrip('\n'))
            disp_arr_sub[i, :, :] = ((temp_disp/255.0) * (max_disp-min_disp) + min_disp)[
                :, :, 0].copy()  # 檢查過沒有disparity value小於0
        if(int(f.readline().rstrip('\n')) == i):
            temp_disp = cv2.imread(
                "warping_dataset/bike/bike_dis/" + str(i) + "_right.png-d.png")
            min_disp = float(f.readline().rstrip('\n'))
            max_disp = float(f.readline().rstrip('\n'))
            disp_arr_mid[i, :, :] = ((temp_disp/255.0) * (max_disp-min_disp) + min_disp)[
                :, :, 0].copy()  # 檢查過沒有disparity value小於0
    disp_arr_sub = np.clip(disp_arr_sub, 0, 255)
    disp_arr_sub = disp_arr_sub.astype(np.uint8)
    disp_arr_mid = np.clip(disp_arr_mid, 0, 255)
    disp_arr_mid = disp_arr_mid.astype(np.uint8)
    f.close()

    return disp_arr_sub, disp_arr_mid

# 計算disparity前二大跟前二小的值屬於哪張disparity map
def disp_max_min(disp_arr, height, width, num):
    disp_max = disp_arr.copy()
    disp_min = disp_arr.copy()
    disp_count_max = np.zeros((num))
    disp_count_min = np.zeros((num))
    for i in range(height):
        for j in range(width):
            if(disp_arr[np.argmax(disp_arr[0:num,i,j]),i,j]==0 and disp_arr[np.argmin(disp_arr[0:num,i,j]),i,j] ==0):
                continue
            else:
                disp_count_max[np.argmax(disp_max[0:num,i,j])] += 1
                disp_max[np.argmax(disp_max[0:num,i,j]), i, j] = 0
                disp_count_max[np.argmax(disp_max[0:num,i,j])] += 1
                
                disp_count_min[np.argmin(disp_min[0:num,i,j])] += 1
                disp_min[np.argmin(disp_min[0:num,i,j]), i, j] = 255
                disp_count_min[np.argmin(disp_min[0:num,i,j])] += 1
                
    disp_count = disp_count_max + disp_count_min
    print("minimum disparity distribution:")
    print(disp_count_min)
    print("MAXimum disparity distribution:")
    print(disp_count_max)
    print("extremum disparity distribution:")     
    print(disp_count)   
        
#用median處理disparity map融合
def outlier_median(disp_arr, height, width):
    disp_arr_median = np.zeros((16, height, width), dtype=np.uint8)
    disp_arr_median[0:8] = disp_arr[0:8].copy()
    disp_arr_median[8:16] = disp_arr[9:17].copy()
    disp_median = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            disp_median[i,j] = np.median(disp_arr_median[:,i,j])
    disp = disp_median.copy()
    disp = disp.astype(np.uint8)
    return disp

#用box_plot處理disparity map融合
def outlier_boxplot(disp_arr, height, width):
    disp_arr_box = np.zeros((16, height, width), dtype=np.uint8)
    disp_arr_box[0:8] = disp_arr[0:8].copy()
    disp_arr_box[8:16] = disp_arr[9:17].copy()
    disp_box = np.zeros((height,width))
    Q1 = int(16/4)
    Q3 = int(16*3/4)
    Q2 = int(16*2/4)
    for i in range(height):
        for j in range(width):
            disp_arr_box[:,i,j].sort()
            q1 = (disp_arr_box[Q1,i,j] + disp_arr_box[Q1+1,i,j])/2
            q3 = (disp_arr_box[Q3,i,j] + disp_arr_box[Q3+1,i,j])/2
            q2 = (disp_arr_box[Q2,i,j] + disp_arr_box[Q2+1,i,j])/2
            IQR = q3-q1
            max_fence = q3 + 1.5*IQR
            min_fence = q1 - 1.5*IQR
            count = 0
            sum = 0
            for k in disp_arr_box[:,i,j]:
                if(k >= min_fence and k <= max_fence):
                    count += 1
                    sum += k
            disp_box[i,j] = sum/count
    disp = disp_box.copy()
    disp = disp.astype(np.uint8)
    return disp

#用outlier_dev處理disparity map融合
def outlier_dev(disp_arr, height, width, multi):
    disp_arr_dev = np.zeros((16, height, width), dtype=np.uint8)
    disp_arr_dev[0:8] = disp_arr[0:8].copy()
    disp_arr_dev[8:16] = disp_arr[9:17].copy()
    disp_dev = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            mean = np.mean(disp_arr_dev[:,i,j])
            stdev = np.std(disp_arr_dev[:,i,j], ddof=0)
            sum = 0
            count = 0
            for k in disp_arr_dev[:,i,j]:
                if((k >= mean-1*stdev) and (k <= mean+(multi*stdev))):
                    sum += k
                    count += 1
            disp_dev[i,j] = sum/count
    disp = disp_dev.copy()
    disp = disp.astype(np.uint8)
    return disp
#用paper去頭去尾方法處理disparity map融合
def outlier_paper(disp_arr, height, width):
    disp = np.zeros((height,width))
    disp_sorted_arr = np.zeros((17,height,width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            temp = np.zeros((17-2-2-1))
            disp_sorted_arr[:,i,j] = sorted(disp_arr[0:17,i,j])
            temp[0:6] = disp_sorted_arr[2:8,i,j]
            temp[6:12] = disp_sorted_arr[9:15,i,j]
            disp[i,j] = np.mean(temp)
    disp = disp.astype(np.uint8)
    return disp

#看每個pixel的disparity值在16張圖中的分布情形
def disp_distribution(disp_arr, height, width):
    wb = Workbook()
    ws = wb.active
    disp_arr_distrib = np.zeros((16, height, width), dtype=np.uint8)
    disp_arr_distrib[0:8] = disp_arr[0:8].copy()
    disp_arr_distrib[8:16] = disp_arr[9:17].copy()
    for i in range(height):
        for j in range(width):
            if(np.max(disp_arr_distrib[:,i,j]) != 0):
                ws.append(disp_arr_distrib[:,i,j].tolist())
    wb.save('warping_dataset/bike/after_compensate/disparity_distribution.xlsx')
    print("done")
#算disparity error越大，是否實際pixel值也差越多
def disp_error_pix_error(disp_arr_mid, disp_arr_sub, height, width):
    downsampled_img = np.zeros((21,height,width,3))    
    for i in range(21):
        downsampled_img[i] = cv2.imread("warping_dataset/bike/LR_data_pair/" + str(i) + "_left.png")
    wb = Workbook()
    ws = wb.active
    error_arr = np.zeros((100,2))
    for k in range(17):
        if( k >= 0 and k <= 3):
            for i in range(height):
                for j in range(width):
                    if((i + disp_arr_mid[k,i,j]) < height):
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i + disp_arr_mid[k,i,j], j]), 0] += np.mean(abs(downsampled_img[k,i,j] - downsampled_img[k,i + disp_arr_mid[k,i,j], j]))
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i + disp_arr_mid[k,i,j], j]), 1] += 1 
        elif( k >= 4 and k <= 7):
            for i in range(height):
                for j in range(width):
                    if((j + disp_arr_mid[k,i,j]) < width):
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i, j + disp_arr_mid[k,i,j]]), 0] += np.mean(abs(downsampled_img[k,i,j] - downsampled_img[k,i, j + disp_arr_mid[k,i,j]]))
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i, j + disp_arr_mid[k,i,j]]), 1] += 1
                        
        elif( k >= 9 and k <= 12):
            for i in range(height):
                for j in range(width):
                    if((j - disp_arr_mid[k,i,j]) >= 0):
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i, j - disp_arr_mid[k,i,j]]), 0] += np.mean(abs(downsampled_img[k,i,j] - downsampled_img[k,i, j - disp_arr_mid[k,i,j]]))
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i, j - disp_arr_mid[k,i,j]]), 1] += 1
                        
        elif( k >= 13 and k <= 16):
            for i in range(height):
                for j in range(width):
                    if((i - disp_arr_mid[k,i,j]) >= 0):
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i - disp_arr_mid[k,i,j], j]), 0] += np.mean(abs(downsampled_img[k,i,j] - downsampled_img[k,i - disp_arr_mid[k,i,j], j]))
                        error_arr[abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i - disp_arr_mid[k,i,j], j]), 1] += 1
    for i in range(100):
        if(error_arr[i,1] != 0):
            error_arr[i,0] = error_arr[i,0] / error_arr[i,1]
        ws.append([error_arr[i,0], error_arr[i,1]])
    wb.save('warping_dataset/bike/error.xlsx')