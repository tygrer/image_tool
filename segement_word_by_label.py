# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from skimage import data,segmentation,measure,morphology,color
import pandas as pd
import pandas
import math



if __name__ == '__main__':
    #demo_dir = '/home/tanggy/Tencent_corrected_for_ctc_0527_181_img_slice/'
    demo_dir = '/home/tanggy/guiyu/TEST/'
    for img in os.listdir(demo_dir):
        image = cv2.imread(demo_dir+img)
        preprocess_txt_crop(demo_dir+img, image)