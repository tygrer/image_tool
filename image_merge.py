import os
import cv2
import numpy as np
dir = "/media/gytang/My Passport3/exp_res/car-detection/car/"
#dir_b = "/media/gytang/My Passport3/RESIDE/synthetic/original"
file_list = os.listdir(dir)
file_list.sort()
# b_list = os.listdir(dir_b)
# b_list.sort()
for _,file in enumerate(file_list):
    for _,imga in enumerate(os.listdir(os.path.join(dir,file))):
        #for b_idx, imgb in enumerate(b_list):
            content_a = cv2.imread(os.path.join(dir, file,imga))
            content_a = cv2.resize(content_a,(256,256))
            # content_b = cv2.imread(os.path.join(dir_b, b_list[a_idx]))
            # content_b = cv2.resize(content_b, (128, 128))
            # content = np.concatenate((content_a,content_b),axis=1)
            if not os.path.exists(os.path.join("/media/gytang/My Passport3/exp_res/car-detection/car-resize",file)):
                os.mkdir(os.path.join("/media/gytang/My Passport3/exp_res/car-detection/car-resize",file))
            cv2.imwrite(os.path.join("/media/gytang/My Passport3/exp_res/car-detection/car-resize",file,imga+".jpg",),content_a)
