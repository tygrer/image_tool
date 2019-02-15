import os
import cv2
import numpy as np
dir_a = "/media/gytang/My Passport/CycleGAN-master5/datasets/D-hazy/trainA.3"
dir_b = "/media/gytang/My Passport/CycleGAN-master5/datasets/D-hazy/trainB.2"
a_list = os.listdir(dir_a)
a_list.sort()
b_list = os.listdir(dir_b)
b_list.sort()
for a_idx, imga in enumerate(a_list):
    #for b_idx, imgb in enumerate(b_list):
        content_a = cv2.imread(os.path.join(dir_a, imga))
        content_a = cv2.resize(content_a,(128,128))
        content_b = cv2.imread(os.path.join(dir_b, b_list[a_idx]))
        content_b = cv2.resize(content_b, (128, 128))
        cv2.imwrite(os.path.join("/media/gytang/My Passport/CycleGAN-master5/datasets/D-hazy/trainA/",imga[:-4]+".jpg"),content_a)
        cv2.imwrite(os.path.join("/media/gytang/My Passport/CycleGAN-master5/datasets/D-hazy/trainB/", b_list[a_idx][:-5]+".jpg"), content_b)
