import cv2 
import numpy as np
import math
from skimage import metrics

def PSNR(Ori_image, Output_image):
    MSE = np.average((Ori_image/1.0-Output_image/1.0)**2)
    if MSE <= 10e-10:
        return 100
    else:
        return 10 * math.log10(255*255/MSE)

def SSIM(Ori_image, Output_image, multichannel=False):
    window_size = 4

    if multichannel:
        Ori_image = cv2.cvtColor(Ori_image, cv2.COLOR_BGR2GRAY)
        Output_image = cv2.cvtColor(Output_image, cv2.COLOR_BGR2GRAY)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1*255)**2
    C2 = (K2*255)**2
    C3 = C2/2

    SSIM_list = []
    for i in range(0,np.shape(Ori_image)[0],window_size):
        for j in range(0,np.shape(Ori_image)[1],window_size):
            Ori_patch = Ori_image[i:i+window_size,j:j+window_size].flatten().astype(np.float64)
            Output_patch = Output_image[i:i+window_size,j:j+window_size].flatten().astype(np.float64)

            Ori_mean = np.mean(Ori_patch)
            Output_mean = np.mean(Output_patch)

            Ori_sigma = np.std(Ori_patch,ddof=1)
            Output_sigma = np.std(Output_patch,ddof=1)

            luminance = ((2*Ori_mean*Output_mean + C1) / 
                            (Ori_mean**2 + Output_mean**2 + C1))

            constrast = ((2*Ori_sigma*Output_sigma + C2) /
                            (Ori_sigma**2 + Output_sigma**2 + C2))

            cov_matrix = np.cov(np.array([Ori_patch,Output_patch]),ddof=1)
            
            structure = ((cov_matrix[0,1] + C3) /
                         (Ori_sigma*Output_sigma + C3))

            SSIM_list.append(luminance*constrast*structure)
        
    SSIM_val = np.mean(SSIM_list)

    return SSIM_val

def NRMSE(Ori_image, Output_image):
    Ori_image.astype(np.float64)
    Output_image.astype(np.float64)
    NRMSE_val = np.sum((Ori_image-Output_image)**2)
    NRMSE_val /= np.sum((Ori_image)**2)
    return np.sqrt(NRMSE_val)



def main():
    a = np.array([[2,2,2,2],
                  [3,3,3,3],
                  [1,2,3,4],
                  [1,1,1,1]])
    
    b = np.array([[8,2,2,2],
                  [8,3,3,3],
                  [8,1,2,3],
                  [8,1,1,1]])
    
    SSIM(a,b)

    # img = cv2.imread('./IMAX/1.bmp')
    # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # h,w,c = np.shape(img)
    #Out_img = cv2.blur(img,(3,3))
    #Out_gray_img = cv2.blur(gray_img,(3,3))
    # Out_img = cv2.resize(img,(h//2,w//2),interpolation=cv2.INTER_CUBIC)
    # Out_img = cv2.resize(Out_img,(h,w),interpolation=cv2.INTER_CUBIC)
    # Out_gray_img = cv2.cvtColor(Out_img,cv2.COLOR_BGR2GRAY)
    #our_psnr = PSNR(img, Out_img)

    # print(NRMSE(img, Out_img))


    # sk_ssim = metrics.structural_similarity(img, Out_img, multichannel=True)
    # print(sk_ssim)
    # SSIM_R = SSIM(img[:,:,2], Out_img[:,:,2])
    # SSIM_G = SSIM(img[:,:,1], Out_img[:,:,1])
    # SSIM_B = SSIM(img[:,:,0], Out_img[:,:,0])
    # SSIM_val = (SSIM_R+SSIM_B+SSIM_G)/3
    # print(SSIM_val)


if __name__ == "__main__":
    main()