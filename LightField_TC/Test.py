# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:17:34 2021

@author: brian

備註：PMS裡面是right補給left

"""

import cv2
import numpy as np
from openpyxl import Workbook
from metrics import SSIM
import utils

height, width = 0,0

def recoverDisValue():
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

def dilateFirst(occlusion_arr,disp_arr_mid,both_occlusion):
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
                    both_occlusion[i, j] = 1
                    # both_occlusion[i, j] = 1
        # displace = cv2.dilate(displace, kernel, iterations=1)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    return disp_arr_mid, both_occlusion

def findOcclusionFirst(occlusion_arr,disp_arr_mid,both_occlusion):
    disp_arr_mid_copy = disp_arr_mid.copy()
    kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion
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
                    both_occlusion[i, j] = 1
        displace = cv2.dilate(displace, kernel, iterations=1)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    return disp_arr_mid, both_occlusion

def diffOcc_dilateFirst(occlusion_arr,disp_arr_mid,both_occlusion):
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
                    both_occlusion[tmp, i, j] = 1
                    # both_occlusion[i, j] = 1
        # displace = cv2.dilate(displace, kernel, iterations=1)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    return disp_arr_mid, both_occlusion

def diffOcc_findOcclusionFirst(occlusion_arr,disp_arr_mid,both_occlusion):
    disp_arr_mid_copy = disp_arr_mid.copy()
    kernel = np.ones((3, 3), np.uint8) #kernel為1, 1 = occlusion
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
                    both_occlusion[tmp, i, j] = 1
        displace = cv2.dilate(displace, kernel, iterations=1)
        for i in range(height):
            for j in range(width):
                if(displace[i, j] == 1):
                    disp_arr_mid[k, i, j] = disp_arr_mid_copy[sym, i, j]
    return disp_arr_mid, both_occlusion

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # 計算中心圖抽樣放大前後誤差diff(為int32有正有負)
    # 計算縮小放大的residual
    print("Calculating difference...")
    mid = cv2.imread("warping_dataset/bike/HD_data/8.png")  # 讀取中心圖
    height = mid.shape[0]
    width = mid.shape[1]
    scale_percent = 1/4  # percent of original size #換圖要改
    mid_down = cv2.resize(mid, (int(height*scale_percent),
                                int(width*scale_percent)), interpolation=cv2.INTER_LINEAR)  # 換圖要改
    mid_re = cv2.resize(mid_down, (width, height),
                        interpolation=cv2.INTER_LINEAR)  # 換圖要改
    dif = (mid.astype(np.int32) - mid_re.astype(np.int32))[:, :, :].copy()
    # 將周圍的圖縮小放大後儲存
    """
    for i in range(17,21):
            right = cv2.imread("warping_dataset/bike/HD_data/8.png")#換圖要改
            right_down = cv2.resize(right, (int(width*scale_percent), int(height*scale_percent)), interpolation = cv2.INTER_LINEAR)#換圖要改
            right_re = cv2.resize(right_down, (width, height), interpolation = cv2.INTER_LINEAR)#換圖要改
            left = cv2.imread("warping_dataset/bike/HD_data/"+ str(i) +".png")#換圖要改
            left_down = cv2.resize(left, (int(height*scale_percent), int(width*scale_percent)), interpolation = cv2.INTER_LINEAR)#換圖要改
            left_re = cv2.resize(left_down, (width, height), interpolation = cv2.INTER_LINEAR)#換圖要改
            cv2.imwrite("warping_dataset/bike/LR_data_pair/" + str(i) + "_right.png", right_re)
            cv2.imwrite("warping_dataset/bike/LR_data_pair/" + str(i) + "_left.png", left_re)

    for i in range(17):
            right = cv2.imread("warping_dataset/bike/HD_data/8.png")#換圖要改
            right_down = cv2.resize(right, (int(width*scale_percent), int(height*scale_percent)), interpolation = cv2.INTER_LINEAR)#換圖要改
            right_re = cv2.resize(right_down, (width, height), interpolation = cv2.INTER_LINEAR)#換圖要改
            left = cv2.imread("warping_dataset/bike/HD_data/"+ str(i) +".png")#換圖要改
            left_down = cv2.resize(left, (int(height*scale_percent), int(width*scale_percent)), interpolation = cv2.INTER_LINEAR)#換圖要改
            left_re = cv2.resize(left_down, (width, height), interpolation = cv2.INTER_LINEAR)#換圖要改
            cv2.imwrite("warping_dataset/bike/LR_data_pair/" + str(i) + "_right.png", right_re)
            cv2.imwrite("warping_dataset/bike/LR_data_pair/" + str(i) + "_left.png", left_re)
    """
    # 把每一個disparity map還原回真實視差值 [0,255]->[]
    disp_arr_sub, disp_arr_mid = recoverDisValue()

    # =============================================================================
    #     #%% 計算disparity前二大跟前二小的值屬於哪張disparity map
    #     print("Calculating MAX and min disparity values...")
    #     utils.disp_max_min(disp_arr_mid, height, width, 17)
    #
    #     #%% 看disparity map值的分布狀態
    #     print("Check diparity distribution...")
    #     utils.disp_distribution(disp_arr_mid, height, width)
    # =============================================================================

    # 找occlusion的區域在哪裡
    print("Occlusion handling...")
    threshold = 1
    dilate_First = False
    diffOcc = False
    occlusion_arr = np.zeros((17, height, width), dtype=np.uint8)
    for k in range(17):
        if(k >= 0 and k <= 3):
            for i in range(height):
                for j in range(width):
                    if((i + disp_arr_mid[k, i, j]) < height):
                        if(abs(disp_arr_mid[k, i, j] - disp_arr_sub[k, i + disp_arr_mid[k, i, j], j]) > threshold):
                            occlusion_arr[k, i, j] = 1
                    else:
                        occlusion_arr[k, i, j] = 1
        elif(k >= 4 and k <= 7):
            for i in range(height):
                for j in range(width):
                    if((j + disp_arr_mid[k, i, j]) < width):
                        if(abs(disp_arr_mid[k, i, j] - disp_arr_sub[k, i, j + disp_arr_mid[k, i, j]]) > threshold):
                            occlusion_arr[k, i, j] = 1
                    else:
                        occlusion_arr[k, i, j] = 1
        elif(k >= 9 and k <= 12):
            for i in range(height):
                for j in range(width):
                    if((j - disp_arr_mid[k, i, j]) >= 0):
                        if(abs(disp_arr_mid[k, i, j] - disp_arr_sub[k, i, j - disp_arr_mid[k, i, j]]) > threshold):
                            occlusion_arr[k, i, j] = 1
                    else:
                        occlusion_arr[k, i, j] = 1
        elif(k >= 13 and k <= 16):
            for i in range(height):
                for j in range(width):
                    if((i - disp_arr_mid[k, i, j]) >= 0):
                        if(abs(disp_arr_mid[k, i, j] - disp_arr_sub[k, i - disp_arr_mid[k, i, j], j]) > threshold):
                            occlusion_arr[k, i, j] = 1
                    else:
                        occlusion_arr[k, i, j] = 1
        # mid_1 = cv2.imread("warping_dataset/bike/HD_data/"+str(k)+".png")
        # print(mid_1.shape)
        # for i in range(height):
        #     for j in range(width):
        #         if occlusion_arr[k,i,j]!=0:
        #             mid_1[i,j,:] = 0
        # cv2.imwrite("warping_dataset/test1/"+ str(k) +".png", mid_1)
    # tmp = occlusion_arr[7]
    # cv2.imshow(str(k),tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 對occlusion區域做displace
    #eg.左邊是occlusion右邊也是=不做, 左邊是occlusion右邊不是=補

    both_occlusion = np.zeros((height, width))
    _occlusion = np.zeros((4,height, width))
    if(dilate_First and diffOcc):
        disp_arr_mid, _occlusion = diffOcc_dilateFirst(occlusion_arr,disp_arr_mid,_occlusion)
        #tmpOcc = both_occlusion[both_occlusion[:,:]==1]=255
        # cv2.imwrite("warping_dataset/output(changeRightSideImage)/bothOcc(dilatFirst).png", both_occlusion)
    elif(diffOcc):#一圈一張
        disp_arr_mid, _occlusion = diffOcc_findOcclusionFirst(occlusion_arr,disp_arr_mid,_occlusion)
        # tmpOcc = both_occlusion[both_occlusion[:,:]==1]=255
        # cv2.imwrite("warping_dataset/output(changeRightSideImage)/bothOcc.png", both_occlusion)
    elif(dilate_First):
        disp_arr_mid, both_occlusion = dilateFirst(occlusion_arr,disp_arr_mid,both_occlusion)
    else:
        disp_arr_mid, both_occlusion = findOcclusionFirst(occlusion_arr,disp_arr_mid,both_occlusion)

    # for i in range(4):
    #     ttmp = _occlusion[i]
    #     cv2.imshow(str(i),ttmp)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     tmpOcc = ttmp[ttmp[:,:]==1]=255
    #     cv2.imwrite("warping_dataset/output(changeRightSideImage)/bothOcc_dF" + str(i) + ".png", ttmp)
        


    # =============================================================================
    #     for i in range(height):
    #         for j in range(width):
    #             if(both_occlusion[i,j] == 1):
    #                 if(occlusion_arr[com_a,i,j] != 255 and occlusion_arr[com_b,i,j] != 255):
    #                     disp_arr_mid[k,i,j] = (disp_arr_mid_copy[com_a,i,j] + disp_arr_mid_copy[com_b,i,j])//2
    #                 elif(occlusion_arr[com_a,i,j] != 255):
    #                     disp_arr_mid[k,i,j] = disp_arr_mid_copy[com_a,i,j]
    #                 elif(occlusion_arr[com_b,i,j] != 255):
    #                     disp_arr_mid[k,i,j] = disp_arr_mid_copy[com_b,i,j]
    #                     #之後補如果四方向全部都occlusion要怎麼處理
    # =============================================================================
    # 選擇如何處理outliers, output:disp
    # 變成一張
    # 測試圖片都是用十字16張圖片，之後有需要再從utils裡面改
    case = 0

    # 用median來處理outlier...0
    if(case == 0):
        print("Remove outliers by median...")
        disp = utils.outlier_median(disp_arr_mid, height, width)
        method = "median"

    # 用盒鬚圖來處理outlier...1
    if(case == 1):
        print("Remove outliers by box plot...")
        disp = utils.outlier_boxplot(disp_arr_mid, height, width)
        method = "boxplot"

    # 用標準差來去除outlier...2
    if(case == 2):
        print("Remove outliers by deviation...")
        disp = utils.outlier_dev(disp_arr_mid, height, width, 1)
        method = "stdev"

    # 將disparity maps在每個pixel中最大跟最小各兩個值刪除後平均(論文方法)...3
    if(case == 3):
        print("Remove outliers by 去頭去尾...")
        disp = utils.outlier_paper(disp_arr_mid, height, width)
        method = "blend"

    # 做基於blended disparity map的warpping
    print("Warping by " + method + " disparity map...")
    wb = Workbook()
    ws = wb.active
    ws.append(["", "bilinear_psnr", "compensate_" + method + "_psnr",
               "bilinear_SSIM", "compensate_" + method + "_SSIM"])
    diftmp = dif.astype(np.float32)
    for k in range(17):
        if(k != 8):
            print("Warping difference map by " + method +
                  " disparity on " + str(k) + "...")
            sub = cv2.imread("warping_dataset/bike/HD_data/" + str(k) + ".png")
            sub_down = cv2.resize(sub, (int(
                width*scale_percent), int(height*scale_percent)), interpolation=cv2.INTER_LINEAR)  # 換圖要改
            sub_re = cv2.resize(sub_down, (width, height),
                                interpolation=cv2.INTER_LINEAR)  # 換圖要改
            # cv2.imwrite("warping_dataset/LR/" + str(k) + ".png", sub_re)

            difUp = cv2.resize(
                diftmp, (int(width*4), int(height*4)), interpolation=cv2.INTER_LINEAR)
            dispUp = cv2.resize(
                disp, (int(width*4), int(height*4)), interpolation=cv2.INTER_LINEAR)
            warpUp = np.zeros((int(width*4), int(height*4), 3))
            new_sub = sub_re.astype(np.int32)
            # ------改blend或own disparity改這邊------
            if(k >= 0 and k <= 3):
                for i in range(height*4):
                    for j in range(width*4):
                        if((i + dispUp[i, j]) < height*4):
                            warpUp[i + dispUp[i, j], j, :] = difUp[i, j, :]

            elif(k >= 4 and k <= 7):
                for i in range(height*4):
                    for j in range(width*4):
                        if((j + dispUp[i, j]) < width*4):
                            warpUp[i, j + dispUp[i, j], :] = difUp[i, j, :]

            elif(k >= 9 and k <= 12):
                for i in range(height*4):
                    for j in range(width*4):
                        if((j - dispUp[i, j]) >= 0):
                            warpUp[i, j - dispUp[i, j], :] = difUp[i, j, :]

            elif(k >= 13 and k <= 16):
                for i in range(height*4):
                    for j in range(width*4):
                        if((i - dispUp[i, j]) >= 0):
                            warpUp[i - dispUp[i, j], j, :] = difUp[i, j, :]

  
            warpDown = cv2.resize(warpUp, (width, height),interpolation=cv2.INTER_LINEAR)
            warped = warpDown.astype(np.int32)
            if(diffOcc):
                if(k == 0 or k == 4 or k == 12 or k == 16):
                    tmp = 0
                elif(k == 1 or k == 5 or k == 11 or k == 15):
                    tmp = 1
                elif(k == 2 or k == 6 or k == 10 or k == 14):
                    tmp = 2
                elif(k == 3 or k == 7 or k == 9 or k == 13):
                    tmp = 3

            # cv2.imwrite("warping_dataset/output(changeRightSideImage)/warpedDiff.png", warped)
            # cv2.imwrite("warping_dataset/output(changeRightSideImage)/diff.png", dif)

                for i in range(height):
                    for j in range(width):
                        if(_occlusion[tmp, i, j] != 1):
                            new_sub[i, j, :] += warped[i, j, :]
            else:
                for i in range(height):
                    for j in range(width):
                        if(both_occlusion[i, j] != 1):
                            new_sub[i, j, :] += warped[i, j, :]
            # warped = warpDown.astype(np.int32)

            # for i in range(height):
            #     for j in range(width):
            #         if(both_occlusion[i, j] != 1):
            #             new_sub[i, j, :] += warped[i, j, :]

            # new_sub += warped
            new_sub = np.clip(new_sub, 0, 255)
            new_sub = new_sub.astype(np.uint8)
            cv2.imwrite("warping_dataset/output(changeRightSideImage)/" + str(k) + ".png", new_sub)

            SSIM_R = SSIM(sub[:, :, 2], sub_re[:, :, 2])
            SSIM_G = SSIM(sub[:, :, 1], sub_re[:, :, 1])
            SSIM_B = SSIM(sub[:, :, 0], sub_re[:, :, 0])
            SSIM_sub_re = (SSIM_R + SSIM_B + SSIM_G)/3

            SSIM_R = SSIM(sub[:, :, 2], new_sub[:, :, 2])
            SSIM_G = SSIM(sub[:, :, 1], new_sub[:, :, 1])
            SSIM_B = SSIM(sub[:, :, 0], new_sub[:, :, 0])
            SSIM_new_sub = (SSIM_R + SSIM_B + SSIM_G)/3

            ws.append([str(k), cv2.PSNR(sub, sub_re), cv2.PSNR(
                sub, new_sub), SSIM_sub_re, SSIM_new_sub])
            print(str(k))
            print("sub_re PSNR: " + str(cv2.PSNR(sub, sub_re)))
            print("new_sub PSNR: " + str(cv2.PSNR(sub, new_sub)))
            print("SSIM sub_re: " + str(SSIM_sub_re))
            print("SSIM new_sub: " + str(SSIM_new_sub))
            #cv2.imwrite("warping_dataset/bike/after_compensate/"+ str(k) + "_" + method +"_withoutOcclusion_twoside_thresh1.png", new_sub)

    wb.save('warping_dataset/bike/' + method +
            '_compensate_t=' + str(threshold)+'.xlsx')

# print("Warping by own disparity map...")
# wb = Workbook()
# ws = wb.active
# ws.append(["","bilinear_psnr","compensate_own_psnr","bilinear_SSIM","compensate_own_SSIM"])
# diftmp = dif.astype(np.float32)
# difUp = cv2.resize(diftmp, (int(width*4), int(height*4)), interpolation=cv2.INTER_LINEAR)
# for k in range(17):
#     if(k!=8):
#         print("Warping difference map by own disparity map on " + str(k) + "...")
#         sub = cv2.imread("warping_dataset/bike/HD_data/"+ str(k) +".png")
#         sub_down = cv2.resize(sub, (int(width*scale_percent), int(height*scale_percent)), interpolation = cv2.INTER_LINEAR)#換圖要改
#         sub_re = cv2.resize(sub_down, (width, height), interpolation = cv2.INTER_LINEAR)#換圖要改
#         new_sub = sub_re.astype(np.int32)
#         warped = np.zeros((int(width*4), int(height*4), 3))
#         #------改blend或own disparity改這邊------       
#         dispUp = cv2.resize(disp_arr_mid[k], (int(width*4), int(height*4)), interpolation=cv2.INTER_LINEAR)
#         warpUp = np.zeros((int(width*4), int(height*4), 3))
#         # ------改blend或own disparity改這邊------
#         if(k >= 0 and k <= 3):
#             for i in range(height*4):
#                 for j in range(width*4):
#                     if((i + dispUp[i, j]) < height*4):
#                         warpUp[i + dispUp[i, j], j, :] = difUp[i, j, :]

#         elif(k >= 4 and k <= 7):
#             for i in range(height*4):
#                 for j in range(width*4):
#                     if((j + dispUp[i, j]) < width*4):
#                         warpUp[i, j + dispUp[i, j], :] = difUp[i, j, :]

#         elif(k >= 9 and k <= 12):
#             for i in range(height*4):
#                 for j in range(width*4):
#                     if((j - dispUp[i, j]) >= 0):
#                         warpUp[i, j - dispUp[i, j], :] = difUp[i, j, :]

#         elif(k >= 13 and k <= 16):
#             for i in range(height*4):
#                 for j in range(width*4):
#                     if((i - dispUp[i, j]) >= 0):
#                         warpUp[i - dispUp[i, j], j, :] = difUp[i, j, :]


#         warpDown = cv2.resize(warpUp, (width, height),interpolation=cv2.INTER_LINEAR)       
#         warped = warpDown.astype(np.int32)
#         for i in range(height):
#             for j in range(width):
#                 if(both_occlusion[i, j] != 1):
#                     new_sub[i, j, :] += warped[i, j, :]  

#         new_sub = np.clip(new_sub, 0, 255)
#         new_sub = new_sub.astype(np.uint8)
        
#         SSIM_R = SSIM(sub[:,:,2], sub_re[:,:,2])
#         SSIM_G = SSIM(sub[:,:,1], sub_re[:,:,1])
#         SSIM_B = SSIM(sub[:,:,0], sub_re[:,:,0])
#         SSIM_sub_re = (SSIM_R+SSIM_B+SSIM_G)/3
        
        
#         SSIM_R = SSIM(sub[:,:,2], new_sub[:,:,2])
#         SSIM_G = SSIM(sub[:,:,1], new_sub[:,:,1])
#         SSIM_B = SSIM(sub[:,:,0], new_sub[:,:,0])
#         SSIM_new_sub = (SSIM_R+SSIM_B+SSIM_G)/3
        
#         ws.append([str(k), cv2.PSNR(sub,sub_re), cv2.PSNR(sub, new_sub), SSIM_sub_re, SSIM_new_sub])
#         print(str(k))
#         print("sub_re PSNR: " + str(cv2.PSNR(sub, sub_re)))
#         print("new_sub PSNR: " + str(cv2.PSNR(sub, new_sub)))
#         print("SSIM sub_re: " + str(SSIM_sub_re))
#         print("SSIM new_sub: " + str(SSIM_new_sub))
#         cv2.imwrite("warping_dataset/bike/after_compensate/"+ str(k) +"_own_withoutOcclusion_twoside_thresh1.png", new_sub) 

# wb.save('warping_dataset/bike/own_compensate.xlsx')  