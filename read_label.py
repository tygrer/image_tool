# -*- coding: UTF-8 -*-
import os
import cv2
from segment_word import preprocess_x_map_crop
import sys


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def analysis_label_txt(txt_file):
    f = open(txt_file)
    lines = f.readlines()
    jpg_ls = []
    rootdir = "/home/tanggy/Tencent_corrected_for_ctc_0527_181_img_slice/"
    label_ls = []

    for line in lines:
        if check_contain_chinese(line):
            ind_x = line.find(":")
            print(line)
            path = line[:ind_x]
            label = line[ind_x+1:]
            jpg_name = os.path.basename(path)
            jpg_ls.append(os.path.join(rootdir, jpg_name))
            label_ls.append(label)

    return jpg_ls, label_ls

if __name__ == '__main__':
    demo_dir = '/home/tanggy/Tencent_corrected_for_ctc_0527_181_img_slice/'
    label_dir = '/home/tanggy/Tencent_train.txt'
    img_dir, label_str = analysis_label_txt(label_dir)
    for i in range(607,len(img_dir)):
        image = cv2.imread(img_dir[i])
        preprocess_x_map_crop(img_dir[i], image)


    # file_spl = path.split('/')
    # file_name = file_spl[len(file_spl) - 1]
    # height, width = image.shape[0], image.shape[1]
    # img = cv2.resize(image,(2*width,2*height))
    # height, width = img.shape[0], img.shape[1]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # from sklearn.externals import joblib
    # model_g = joblib.load("thresh.pkl")
    #
    # hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])
    # mid = round(model_g.predict(np.transpose(hist))[0])
    # (_, thresh) = cv2.threshold(gray, mid, gray.max(), cv2.THRESH_BINARY)
    # #cv2.imshow('', thresh)
    # #cv2.waitKey(10)
    # #cv2.imshow('', thresh)
    # #cv2.waitKey(0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # closed = cv2.erode(closed, None, iterations=1)
    # cv2.imshow('', closed)
    # cv2.waitKey(1500)
    #label_image = measure.label(closed)

    # image_label_overlay = color.label2rgb(label_image+1, image=img)
    #
    # cv2.imshow('', image_label_overlay)
    # cv2.waitKey(1500)

    #_, cnts, _ = cv2.findContours((label_image).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # label_image.max()
    # rect = cv2.minAreaRect(i)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)

        # x1 = min(box[:, 1])
        # x2 = max(box[:, 1])
        # y1 = min(box[:, 0])
        # y2 = max(box[:, 0])
        # width = y2 - y1
        # height = x2 - x1
        # area = height*width

    # gray.min()
    # thresh1 = cv2.adaptiveThreshold(gray, gray.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)5
    #cv2.imshow('', img)
    #cv2.waitKey(0)
    #result_img_crop_path = result_img_crop_path + str(i) + file_name
    #result_img_crop_path = '/home/tanggy/guiyu/crop_segmentation/'
    #if not os.path.isdir(result_img_crop_path):
        #os.makedirs(result_img_crop_path)
    #cv2.imwrite(result_img_crop_path, cropImg)
    #
    # (_, cnts, _) = cv2.findContours((label_image).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # c = sorted(cnts, key=cv2.contourArea, reverse=True)