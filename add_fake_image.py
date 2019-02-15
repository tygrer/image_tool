import os
import cv2
import numpy as np
import random
import copy
from skimage import data,segmentation,measure,morphology,color


# def get_crop_word_and_save(box,img,crop_stack):
#
#     x1 = min(box[:,0])
#     x2 = max(box[:,0])
#     y1 = min(box[:,1])
#     y2 = max(box[:,1])
#     #height = y2 - y1
#     width = x2 - x1
#     cv2.(img, (x1, 0), (x1, img.shape[1]), (0, 255, 0), 2)
#     cv2.line(img, (x2, 0), (x2, img.shape[1]), (0, 255, 0), 2)
#     if width > 15:
#         crop_stack.append(width)
#         cv2.imshow('', img)
#         cv2.waitKey(800)
#     return crop_stack
dir_image='/media/gytang/My Passport3/project/trnet/cropresult'

zhang= cv2.imread('/media/gytang/My Passport3/zhang1.png')
zh,zw,_= zhang.shape
zhang = cv2.GaussianBlur(zhang, ksize=(5,5), sigmaX=0, sigmaY=0)
cv2.imshow("nnff", zhang)
cv2.waitKey(1000)
cv2.imshow("", zhang)
cv2.waitKey(1000)
(_, thresh) = cv2.threshold(zhang[:,:,0], 215, zhang[:,:,0].max(), cv2.THRESH_BINARY)
idx_lst = []
cv2.imshow("1", thresh)
cv2.waitKey(1000)
for i in range(thresh.shape[0]):
    for j in range(thresh.shape[1]):
        if thresh[i][j] == 0:
            idx_lst.append([i,j])
# new_img = np.ones((500,500,3))*255
# for i_l in idx_lst:
#     new_img[50+i_l[0],50+i_l[1]] = 0.3*new_img[50+i_l[0],50+i_l[1]] + 0.7*zhang[i_l[0],i_l[1]]
    #print(new_img[50+i_l[0],50+i_l[1]],"->>>",zhang[i_l[0],i_l[1]])
    # new_img[50 + i_l[0], 50 + i_l[1], 1] = int(zhang[i_l[0], i_l[1],1])
    # new_img[50 + i_l[0], 50 + i_l[1], 2] = int(zhang[i_l[0], i_l[1],2])
# cv2.imshow('new',new_img/255)
# cv2.waitKey(0)

#
for i in os.listdir(dir_image):

     image= cv2.imread(os.path.join(dir_image,i))
     h,w,c = image.shape
     new_img = np.ones((zh*2+h-10, w+zw*2-10, 3)) * 255
     nh = int(random.random()*(zh+h-30))
     nw = int(random.random()*(w+zw-30))
     ph = int((zh*2+h-10)/2 - h/2)
     pw = int((w+zw*2-10)/2 - w/2)
     new_img[ph:(ph + h), pw:(pw + w)] = image
     zh_p = np.array([0,0,0])
     alpha = random.uniform(0,1)
     for i_l in idx_lst:
         zh_p[:] = zhang[i_l[0], i_l[1],:]
         blue = (1 - alpha) * new_img[nh + i_l[0], nw + i_l[1]] + alpha * zh_p[0] - 20
         green = (1 - alpha) * new_img[nh + i_l[0], nw + i_l[1]] + alpha * zh_p[1] - 20
         rd_color = random.uniform(0, min(max(blue.flatten()), max(green.flatten())))
     for i_l in idx_lst:
         zh_p[:] = zhang[i_l[0], i_l[1],:]
         if zh_p[0]>30 and zh_p[1]>30:
             zh_p[0] = zh_p[0] - rd_color
             zh_p[1] = zh_p[1] - rd_color
             print(alpha,rd_color,zh_p,zhang[i_l[0], i_l[1]])
         new_img[nh + i_l[0], nw + i_l[1]] = (1-alpha)*new_img[nh+i_l[0],nw+i_l[1]]+alpha*zh_p
     cv2.imshow("nnff", new_img / 255)
     cv2.waitKey(1000)
     crop_image = new_img[ph:(ph + h), pw:(pw + w)]
     cv2.imshow("nnffee", crop_image / 255)
     cv2.waitKey(1000)
