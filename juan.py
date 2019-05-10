import os, cv2
import numpy as np
from rotation_by_houghline import draw_line
img_dir = '/home/gytang/juan_crop/'

for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img)
    imgr = cv2.imread(img_path)
    gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    #(_, thresh) = cv2.threshold(gray, 100, gray.max(), cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #closed = cv2.dilate(gray, None, iterations=3)
    closed = cv2.erode(gray, None, iterations=1)
    closed = 255-closed
    # edges = cv2.Canny(closed, 50, 200, apertureSize=3)
    # cv2.imshow("rotation:", edges)
    # cv2.waitKey(2000)

    #hst_lst = closed[:, :5].mean(axis=-1)
    # cv2.imshow("rotation:", hst_lst)
    # cv2.waitKey(2000)
    (_, thresh) = cv2.threshold(closed, 30, closed.max(), cv2.THRESH_BINARY)
    cv2.imshow("rotation:", thresh)
    cv2.waitKey(2000)
    hed_lst = closed[:, -5:].mean(axis=-1)


    # lines = cv2.HoughLines(edges, 1, np.pi / 180,60)
    # cv2.imshow("",closed)
    # cv2.waitKey(1000)
    # draw_img = draw_line(lines, imgr, True, (0, 255, 0))
    # #draw_img = draw_line(aline, draw_img, True, (0, 0, 255))
    # cv2.imshow("rotation:", draw_img)
    # cv2.waitKey(200)

