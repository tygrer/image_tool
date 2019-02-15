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
crop_stack = []
def map_to_x_dimension(closed):
    xmap = np.amin(closed, axis=0)
    x_begin = []
    x_end = []
    for i in range(len(xmap)):
        if i == 0 and xmap[i] == 0:
            x_begin.append(i)
        if xmap[i] and xmap[i+1] == 0:
            x_begin.append(i)
        if xmap[i] == 0 and xmap[i+1]:
            x_end.append(i)
        if i == len(xmap) - 2:
            if len(x_begin) != len(x_end):
                x_end.append(i+1)
            break
    return x_begin, x_end

#def filter_invalid_image(cropImg):
def get_crop_word_and_save(box,img):

    x1 = min(box[:,0])
    x2 = max(box[:,0])
    y1 = min(box[:,1])
    y2 = max(box[:,1])
    #height = y2 - y1
    width = x2 - x1
    cv2.line(img, (x1, 0), (x1, img.shape[1]), (0, 255, 0), 2)
    cv2.line(img, (x2, 0), (x2, img.shape[1]), (0, 255, 0), 2)
    if width > 15:
        crop_stack.append(width)
    #cv2.imshow('', img)
    #cv2.waitKey(800)

def preprocess_x_map_crop(path,image):
    file_spl = path.split('/')
    file_name = file_spl[len(file_spl) - 1]
    height, width = image.shape[0], image.shape[1]
    img = cv2.resize(image,(width, height))
    height, width = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('', gray)
    #cv2.waitKey(1000)
    #print(gray.min())
    #print(gray.max())
    from sklearn.externals import joblib
    model_g = joblib.load("thresh.pkl")

    hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])
    mid = round(model_g.predict(np.transpose(hist))[0])
    #print(mid)
    (_, thresh) = cv2.threshold(gray, mid, gray.max(), cv2.THRESH_BINARY)
    #cv2.imshow('', thresh)
    #cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=1)
    #closed = cv2.dilate(closed, None, iterations=1)
    #cv2.imshow('', closed)
    #cv2.waitKey(200)
    x_begin, x_end = map_to_x_dimension(closed)

    for i in range(len(x_begin)):
        crop_img = img[:, x_begin[i]:x_end[i],:]
        crop_closed = closed[:, x_begin[i]:x_end[i]]
        box = np.array([[x_begin[i], 0], [x_begin[i],img.shape[1]],[x_end[i], 0], [x_end[i], img.shape[1]]])
        area = crop_img.shape[0]*crop_img.shape[1]

        if area < 200:
            continue
        elif crop_img.shape[0]/crop_img.shape[1] < 0.85:
            #cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
            # cv2.imwrite(filepath_color_label_save, cropImg)
            #     row, col = np.where(label_image == i)
            #cv2.imshow('', img)
            #cv2.waitKey(1000)
            preprocess_label_crop(crop_img)
        else:
            #cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
            #cv2.imshow('', img)
            #cv2.waitKey(500)
            get_crop_word_and_save(box, img)
        # cv2.imwrite(filepath_color_label_save, cropImg)
        #     row, col = np.where(label_image == i)
    #cv2.imshow('', img)
    #cv2.waitKey(1000)
    cv2.imwrite(os.path.join("/home/tanggy/cropimage/",file_name), img)
    crop_stack = []
        #             #row, col = np.where(label_image == i)
        #             #cv2.imshow('', img)
        #             #cv2.waitKey(1000)


#def preprocess_label_crop(path,image):
def preprocess_label_crop(img):
    from sklearn.externals import joblib
    model_g = joblib.load("thresh.pkl")
    hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
    mid = round(model_g.predict(np.transpose(hist))[0])
    #print(mid)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray, mid, gray.max(), cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=1)
    closed = cv2.erode(closed, None, iterations=1)
    #cv2.imshow('', closed)
    #cv2.waitKey(100)
    closed = 255-closed
    label_image = measure.label(closed)
    image_label_overlay = color.label2rgb(label_image, image=gray)

    #cv2.imshow('', image_label_overlay)
    #cv2.waitKey(500)

    #label_image = label_image.max() - label_image

    # compute the rotated bounding box of the largest contour
    for i in range(label_image.min(),label_image.max()+1):
        y,x = np.where(label_image == i)
        area = x.shape[0]/(img.shape[0]*img.shape[1])
        box_x_i = x.min()
        box_x_x = x.max()
        box_y_i = y.min()
        box_y_x = y.max()
        width = box_x_x - box_x_i
        if area<0.4 and area>0.015 and width > 9 and width/img.shape[1] < 1:

            box = np.array([[box_x_i,0],[box_x_i, img.shape[1]], [box_x_x,0],[box_x_x,img.shape[1]]])
            get_crop_word_and_save(box, img)
        elif width/img.shape[1] >= 1:

            if crop_stack is not []:
                def get_median(data):
                    data.sort()

                    half = len(data) // 2
                    return (data[half] + data[~half]) / 2
                fake_width = int(round(get_median(crop_stack)))
                for j in range(int(round(width/fake_width))):
                    box = np.array([[(j*fake_width), 0], [(j*fake_width), img.shape[1]],\
                                    [min(((j+1)*fake_width),img.shape[0]), 0], [min(((j+1)*fake_width),img.shape[0]),\
                                                                                   img.shape[1]]])
                    cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
                    cv2.imshow('', img)
                    cv2.waitKey(500)
                    get_crop_word_and_save(box, img)
            else:
                box = np.array([[box_x_i,0],[box_x_i, img.shape[1]], [box_x_x,0],[box_x_x,img.shape[1]]])
                get_crop_word_and_save(box, img)
            #cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
            #cv2.imshow('', img)
            #cv2.waitKey(500)


if __name__ == '__main__':
    demo_dir = '/home/tanggy/Tencent_corrected_for_ctc_0527_181_img_slice/'
    #demo_dir = '/home/tanggy/guiyu/TEST/'
    for i in range(705, len(os.listdir(demo_dir)), 1):
    #for img in os.listdir(demo_dir):
        image = cv2.imread(demo_dir+os.listdir(demo_dir)[i])
        preprocess_x_map_crop(demo_dir+os.listdir(demo_dir)[i], image)