# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:42:29 2021

@author: brian
"""

import numpy as np
import cv2

def occlusion_Fusion(disp_arr_mid, disp_arr_sub, height, width):
    print("Occlusion handling by fusion...")
    error_arr = np.zeros((17,height,width))
    for k in range(17):
        if( k >= 0 and k <= 3):
            for i in range(height):
                for j in range(width):
                    if((i + disp_arr_mid[k,i,j]) < height):
                        error_arr[k,i,j] = abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i + disp_arr_mid[k,i,j],j])
                        
        elif( k >= 4 and k <= 7):
            for i in range(height):
                for j in range(width):
                    if((j + disp_arr_mid[k,i,j]) < width):
                        error_arr[k,i,j] = abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i,j + disp_arr_mid[k,i,j]])
                        
        elif( k >= 9 and k <= 12):
            for i in range(height):
                for j in range(width):
                    if((j - disp_arr_mid[k,i,j]) >= 0):
                        error_arr[k,i,j] = abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i,j - disp_arr_mid[k,i,j]])
                        
        elif( k >= 13 and k <= 16):
            for i in range(height):
                for j in range(width):
                    if((i - disp_arr_mid[k,i,j]) >= 0):
                        error_arr[k,i,j] = abs(disp_arr_mid[k,i,j] - disp_arr_sub[k,i - disp_arr_mid[k,i,j],j])
    
    disp_arr_mid_copy = disp_arr_mid.copy()
    inde_occ_arr = np.zeros((17,height,width), dtype = np.uint8)#其他三個的error都比自己大的時候
    for k in range(17):
        sym = 16-k
        if(k<8):
            if(k<4):
                com_a = k+4
            else:
                com_a = k-4
        else:
            if(k>12):
                com_a = k-4
            else:
                com_a = k+4
        com_b = 16 - com_a
        for i in range(height):
            for j in range(width):
                deno = 0
                if(error_arr[k,i,j] > 1):#如果自己的disparity是occlusion
                    if(error_arr[sym,i,j] == 0):#對面若error=0拿對面
                        disp_arr_mid[k,i,j] = disp_arr_mid_copy[sym,i,j]
                    else:#否則看另外兩邊
                        if(error_arr[com_a,i,j] == 0 and error_arr[com_b,i,j] == 0):
                            disp_arr_mid[k,i,j] = (disp_arr_mid_copy[com_a,i,j] + disp_arr_mid_copy[com_b,i,j])//2
                        elif(error_arr[com_a,i,j] == 0):
                            disp_arr_mid[k,i,j] = disp_arr_mid_copy[com_a,i,j]
                        elif(error_arr[com_b,i,j] == 0):
                            disp_arr_mid[k,i,j] = disp_arr_mid_copy[com_b,i,j]
                        else: #另外三張的error都不等於0，把disparity error小於自己那幾張拿來做fusion
                            if(error_arr[sym,i,j] < error_arr[k,i,j]):
                                sym_error = 1/error_arr[sym,i,j]
                                deno += sym_error
                            else:
                                sym_error = 0
                            if(error_arr[com_a,i,j] < error_arr[k,i,j]):
                                com_a_error = 1/error_arr[com_a,i,j]
                                deno += com_a_error
                            else:
                                com_a_error = 0
                            if(error_arr[com_b,i,j] < error_arr[k,i,j]):
                                com_b_error = 1/error_arr[com_b,i,j]
                                deno += com_b_error
                            else:
                                com_b_error = 0 
                            if(deno != 0):
                                disp_arr_mid[k,i,j] = ((sym_error/deno) * disp_arr_mid_copy[sym,i,j]) + ((com_a_error/deno) * disp_arr_mid_copy[com_a,i,j]) + ((com_b_error/deno) * disp_arr_mid_copy[com_b,i,j])
                            else:
                                inde_occ_arr[k,i,j] = 1#全部都是occlusion的就不補
    return(disp_arr_mid, inde_occ_arr)
                          
def oneOcc_dilateFirst(occlusion_arr, disp_arr_mid, height, width):
    one_occlusion = np.zeros((height, width))
    disp_arr_mid_copy = disp_arr_mid.copy()
    kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion    
    for k in range(17):
        occlusion_arr[k] = cv2.dilate(occlusion_arr[k],kernel,iterations = 1)
    for k in range(17):
        sym = 16-k
        if(k < 8):
            if(k < 4):
                com_a = k+4
            else:
                com_a = k-4
        else:
            if(k > 12):
                com_a = k-4
            else:
                com_a = k+4
        com_b = 16 - com_a

        # both_occlusion = np.zeros((height, width))
        displace = occlusion_arr[k].copy()  # 要用對面補的區域
        for i in range(height):
            for j in range(width):
                # 對稱圖也同時為occlusion
                if(occlusion_arr[k, i, j] == occlusion_arr[sym, i, j] and occlusion_arr[k, i, j] == 1):
                    displace[i, j] = 0
                    one_occlusion[i, j] = 1
                    # both_occlusion[i, j] = 1
        # displace = cv2.dilate(displace, kernel, iterations=1)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    count = 0
    for i in range(height):
        for j in range(width):
            if(one_occlusion[i,j] == 1):
                count += 1
    print(count)
    return disp_arr_mid, one_occlusion

# =============================================================================
# def dilateAfter(occlusion_arr,disp_arr_mid, height, width):
#     both_occlusion = np.zeros((height, width))
#     disp_arr_mid_copy = disp_arr_mid.copy()
#     kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion
#     for k in range(17):
#         sym = 16-k
#         if(k < 8):
#             if(k < 4):
#                 com_a = k+4
#             else:
#                 com_a = k-4
#         else:
#             if(k > 12):
#                 com_a = k-4
#             else:
#                 com_a = k+4
#         com_b = 16 - com_a
# 
#         # both_occlusion = np.zeros((height, width))
#         displace = occlusion_arr[k].copy()  # 要用對面補的區域
#         for i in range(height):
#             for j in range(width):
#                 # 對稱圖也同時為occlusion
#                 if(occlusion_arr[k, i, j] == occlusion_arr[sym, i, j] and occlusion_arr[k, i, j] == 1):
#                     displace[i, j] = 0
#                     both_occlusion[i, j] = 1
#         displace = cv2.dilate(displace, kernel, iterations=1)
#         for i in range(height):
#             for j in range(width):
#                 if(displace[i, j] == 1):
#                     disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
#     return disp_arr_mid, both_occlusion
# =============================================================================

def ringOcc_dilateFirst(occlusion_arr,disp_arr_mid, height, width):
    ring_occlusion = np.zeros((4,height, width))
    disp_arr_mid_copy = disp_arr_mid.copy()
    kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion
    for k in range(17):
        occlusion_arr[k] = cv2.dilate(occlusion_arr[k],kernel,iterations = 1)
    for k in range(17):
        sym = 16-k
        if(k < 8):
            if(k < 4):
                com_a = k+4
            else:
                com_a = k-4
        else:
            if(k > 12):
                com_a = k-4
            else:
                com_a = k+4
        com_b = 16 - com_a

        if(k == 0 or k == 4 or k == 12 or k == 16):
            tmp = 0
        elif(k == 1 or k == 5 or k == 11 or k == 15):
            tmp = 1
        elif(k == 2 or k == 6 or k == 10 or k == 14):
            tmp = 2
        elif(k == 3 or k == 7 or k == 9 or k == 13):
            tmp = 3

        # both_occlusion = np.zeros((height, width))
        displace = occlusion_arr[k].copy()  # 要用對面補的區域
        for i in range(height):
            for j in range(width):
                # 對稱圖也同時為occlusion
                if(occlusion_arr[k, i, j] == occlusion_arr[sym, i, j] and occlusion_arr[k, i, j] == 1):
                    displace[i, j] = 0
                    ring_occlusion[tmp, i, j] = 1
        # displace = cv2.dilate(displace, kernel, iterations=1)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    count = np.zeros((4))
    for a in range(4):
        for b in range(height):
            for c in range(width):
                if(ring_occlusion[a,b,c] == 1):
                    count[a] += 1
    print(count)
    return disp_arr_mid, ring_occlusion

# =============================================================================
# def diffOcc_dilateAfter(occlusion_arr,disp_arr_mid, height, width):
#     _occlusion = np.zeros((4,height, width))
#     disp_arr_mid_copy = disp_arr_mid.copy()
#     kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion
#     for k in range(17):
#         sym = 16-k
#         if(k < 8):
#             if(k < 4):
#                 com_a = k+4
#             else:
#                 com_a = k-4
#         else:
#             if(k > 12):
#                 com_a = k-4
#             else:
#                 com_a = k+4
#         com_b = 16 - com_a
#         if(k == 0 or k == 4 or k == 12 or k == 16):
#             tmp = 0
#         elif(k == 1 or k == 5 or k == 11 or k == 15):
#             tmp = 1
#         elif(k == 2 or k == 6 or k == 10 or k == 14):
#             tmp = 2
#         elif(k == 3 or k == 7 or k == 9 or k == 13):
#             tmp = 3
#         # both_occlusion = np.zeros((height, width))
#         displace = occlusion_arr[k].copy()  # 要用對面補的區域
#         for i in range(height):
#             for j in range(width):
#                 # 對稱圖也同時為occlusion
#                 if(occlusion_arr[k, i, j] == occlusion_arr[sym, i, j] and occlusion_arr[k, i, j] == 1):
#                     displace[i, j] = 0
#                     _occlusion[tmp, i, j] = 1
#         displace = cv2.dilate(displace, kernel, iterations=1)
#         for i in range(height):
#             for j in range(width):
#                 if(displace[i, j] == 1):
#                     disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
#     return disp_arr_mid, _occlusion
# =============================================================================
def indeOcc_dilateFirst(occlusion_arr,disp_arr_mid, height, width):
    inde_occlusion = np.zeros((17, height, width))
    disp_arr_mid_copy = disp_arr_mid.copy()
    kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion
    for k in range(17):
        occlusion_arr[k] = cv2.dilate(occlusion_arr[k],kernel,iterations = 1)
    for k in range(17):
        sym = 16-k
        if(k < 8):
            if(k < 4):
                com_a = k+4
            else:
                com_a = k-4
        else:
            if(k > 12):
                com_a = k-4
            else:
                com_a = k+4
        com_b = 16 - com_a

        displace = occlusion_arr[k].copy()  # 要用對面補的區域
        count = 0
        for i in range(height):
            for j in range(width):
                # 對稱圖也同時為occlusion
                if(occlusion_arr[k, i, j] == occlusion_arr[sym, i, j] and occlusion_arr[k, i, j] == 1):
                    displace[i, j] = 0
                    inde_occlusion[k, i, j] = 1
                    count += 1
        print(count)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    return disp_arr_mid, inde_occlusion