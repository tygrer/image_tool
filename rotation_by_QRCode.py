from rotation_by_houghline import draw_line_rotation_main,rotate,alter_HoughLines,draw_line
import numpy as np
import pickle
import os
#from skimage import data,segmentation,measure,morphology,color
import cv2
import math
def rotation_axis(xmin,ymin,xmax,ymax,angle,img,rotated_img,flage,scale=0.5):

    '''
    x′=(x−x0)∗cos(p)+(y−y0)∗sin(p)+x0
    y′=-(x−x0)∗sin(p)+(y−y0)∗cos(p)+y0
    :param ymin:
    :param ymax:
    :param xmin:
    :param xmax:
    :param angle:
    :param img:
    :param rotated_img:
    :param flage:
    :return:
    '''
    print("***********angle:", angle)
    #angle += np.pi/90
    (h,w) = img.shape[:2]
    centr_h = img.shape[0] / 2
    centr_w = img.shape[1] / 2

    angle = angle*180/np.pi
    M = cv2.getRotationMatrix2D((centr_w,centr_h), angle, 1)
    #print("s1:", s, "t1:", t)
    #print("w:", w, "h:", h)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    print(M[0,0],M[0,1])
    print(np.sin(-angle),np.cos(-angle))
    nw = int((h*sin)+(w*cos))
    nh = int((h*cos)+(w*sin))
    M[0,2] += (nw/2) - w/2
    M[1,2] += (nh/2) - h/2
    points = np.hstack((xmin,ymin,xmax,ymin,xmin,ymax,xmax,ymax)).reshape((-1,2))
    points = np.hstack((points,np.ones((len(points),1))))
    rotated_points = M.dot(points.T).T
    x,y = np.split(rotated_points, 2, axis=1)
    x,y = x.reshape((-1, 4)), y.reshape((-1, 4))
    (h, w) = rotated_img.shape[:2]
    xmin = x.min(axis=1,keepdims=True)
    xmax = x.max(axis=1,keepdims=True)
    ymin = y.min(axis=1,keepdims=True)
    ymax = y.max(axis=1,keepdims=True)
    zeros, ones = np.zeros_like(xmin), np.ones_like(xmin)
    xmin = np.maximum(xmin, zeros)
    xmax = np.minimum(xmax, w)
    ymin = np.maximum(ymin, zeros)
    ymax = np.minimum(ymax, h)

    print("xmin:",xmin,"xmax:",xmax,"ymin:",ymin,"ymax:",ymax)

    if flage:
        scale = 0.5
        rotated_img = cv2.resize(rotated_img, (int(rotated_img.shape[1] * scale), int(rotated_img.shape[0] * scale)))
        cv2.rectangle(rotated_img, (int(xmin*scale),int(ymin*scale)),(int(xmax*scale),int(ymax*scale)), (0, 255, 0), 2)

        cv2.imshow("", rotated_img)
        cv2.waitKey(2000)
    return [int(xmin[0,0]),int(ymin[0,0]),int(xmax[0,0]),int(ymax[0,0])]
def get_rotation_axis_by_cv2(ymin,ymax,xmin,xmax,angle,img,rotated_img,flage,scale=0.5):
    axis_lst = []
    (h, w) = img.shape[:2]
    centr_h = img.shape[0] / 2
    centr_w = img.shape[1] / 2

    angle = angle * 180 / np.pi

    pLTN.x = pLTx * np.cos(angle) + pLTy * np.sin(angle)
    pLTN.y = -pLTx * np.sin(angle) + pLTy * np.cos(angle)

    pLTx = -w / 2
    pLTy = h / 2

    pRTx = w / 2
    pRTy = h / 2

    pLBx = -w/ 2
    pLBy = -h / 23

    pRBx = w / 2
    pRBy = -h / 2

    cosa = np.cos(angle)
    sina = np.sin(angle)
    pLTNx = pLTx * cosa + pLTy * sina
    pLTNy = -pLTx * sina + pLTy * cosa
    pRTNx = pRTx * cosa + pRTy * sina
    pRTNy = -pRTx * sina + pRTy * cosa
    pLBNx = pLBx * cosa + pLBy * sina
    pLBNy = -pLBx * sina + pLBy * cosa
    pRBNx = pRBx * cosa + pRBy * sina
    pRBNy = -pRBx * sina + pRBy * cosa
    s = max(abs(pRTNx - pLBNx), abs(pLTNx - pRBNx))
    t = max(abs(pRTNy - pLBNy), abs(pLTNy - pRBNy))
    print("cos angle:",np.sin(angle),"sin angle:",np.cos(angle))
    s1 = ((0 - w/2) * np.cos(angle)) + ((0 - h/2) * np.sin(angle))
    t1 = (-(0 - w/2) * np.sin(angle)) + ((0 - h/2) * np.cos(angle))
    s2 = ((w - w/2) * np.cos(angle)) - ((0 - h/2) * np.sin(angle))
    t2 = ((w - w/2) * np.sin(angle)) + ((0 - h/2) * np.cos(angle))
    s3 = ((w - w/2) * np.cos(angle)) - ((h - h/2) * np.sin(angle))
    t3 = ((w - w/2) * np.sin(angle)) + ((h - h/2) * np.cos(angle))
    s4 = ((0 - w/2) * np.cos(angle)) + ((h - h/2) * np.sin(angle))
    t4 = (-(0 - w/2) * np.sin(angle)) + ((h - h/2) * np.cos(angle))
    print(s1,", ",t1)
    print(s2, ", ", t2)
    print(s3, ", ", t3)
    print(s4, ", ", t4)

    s = max(abs(s4 - s1), abs(s3 - s2))
    t = max(abs(t4 - t1), abs(t3 - t2))
    s1 = (abs(w - w/2) * np.cos(angle)) - ((h - h/2) * np.sin(angle)) + w/2
    t1 = (abs(w - w/2) * np.sin(angle)) + ((h - h/2) * np.cos(angle)) + h/2
    angle = angle*180/np.pi
    s1 = ((xmin - centr_w) * np.cos(angle)) + ((ymin - centr_h) * np.sin(angle)) + rotated_img.shape[1]/2
    t1 = -((xmin - centr_w) * np.sin(angle)) + ((ymin - centr_h) * np.cos(angle)) + rotated_img.shape[0]/2
    axis_lst.append([s1,t1])
    s2 = ((xmax - centr_w) * np.cos(angle)) + ((ymin - centr_h) * np.sin(angle)) + rotated_img.shape[1]/2
    t2 = -((xmax - centr_w) * np.sin(angle)) + ((ymin - centr_h) * np.cos(angle)) + rotated_img.shape[0]/2
    axis_lst.append([s2,t2])
    s3 = ((xmin - centr_w) * np.cos(angle)) + ((ymax - centr_h) * np.sin(angle)) + rotated_img.shape[1]/2
    t3 = -((xmin - centr_w) * np.sin(angle)) + ((ymax - centr_h) * np.cos(angle)) + rotated_img.shape[0]/2
    axis_lst.append([s3,t3])
    s4 = ((xmax - centr_w) * np.cos(angle)) + ((ymax - centr_h) * np.sin(angle)) + rotated_img.shape[1]/2
    t4 = -((xmax - centr_w) * np.sin(angle)) + ((ymax - centr_h) * np.cos(angle)) + rotated_img.shape[0]/2

    rotated_boxes = np.hstack((ymin,xmin,ymax,xmax))

    s1 = ((xmin - centr_w) * np.cos(angle)) - ((ymin - centr_h) * np.sin(angle)) + centr_w
    t1 = ((xmin - centr_w) * np.sin(angle)) + ((ymin - centr_h) * np.cos(angle)) + centr_h
    axis_lst.append([s1,t1])
    s2 = ((xmax - centr_w) * np.cos(angle)) - ((ymin - centr_h) * np.sin(angle)) + centr_w
    t2 = ((xmax - centr_w) * np.sin(angle)) + ((ymin - centr_h) * np.cos(angle)) + centr_h
    axis_lst.append([s2,t2])
    s3 = ((xmin - centr_w) * np.cos(angle)) - ((ymax - centr_h) * np.sin(angle)) + centr_w
    t3 = ((xmin - centr_w) * np.sin(angle)) + ((ymax - centr_h) * np.cos(angle)) + centr_h
    axis_lst.append([s3,t3])
    s4 = ((xmax - centr_w) * np.cos(angle)) - ((ymax - centr_h) * np.sin(angle)) + centr_w
    t4 = ((xmax - centr_w) * np.sin(angle)) + ((ymax - centr_h) * np.cos(angle)) + centr_h
    axis_lst.append([s4,t4])

    xmin = int(max(min(s1,s2,s3,s4),0)*scale)
    ymin = int(max(min(t1,t2,t3,t4),0)*scale)
    xmax = min(int(max(s1,s2,s3,s4)*scale),rotated_img.shape[1])
    ymax = min(int(max(t1,t2,t3,t4)*scale),rotated_img.shape[0])
    print("rotation_img",rotated_img.shape)
    if flage:
        scale = 0.5
        rotated_img = cv2.resize(rotated_img, (int(rotated_img.shape[1] * scale), int(rotated_img.shape[0] * scale)))
        cv2.line(rotated_img,  (int(s2*scale), int(t2*scale)), (int(s1*scale), int(t1*scale)),(0, 0, 255))
        print(int(s2*scale), int(t2*scale), int(s1*scale), int(t1*scale))
        cv2.imshow("", rotated_img)
        cv2.waitKey(1000)
        cv2.line(rotated_img,  (int(s4*scale), int(t4*scale)), (int(s2*scale), int(t2*scale)),(0, 0, 255))
        print(int(s4*scale), int(t4*scale), int(s2*scale), int(t2*scale))
        cv2.imshow("", rotated_img)
        cv2.waitKey(1000)
        cv2.line(rotated_img,  (int(s3*scale), int(t3*scale)),(int(s4*scale), int(t4*scale)), (0, 0, 255))
        print(int(s3*scale), int(t3*scale),int(s4*scale), int(t4*scale))
        cv2.imshow("", rotated_img)
        cv2.waitKey(1000)
        cv2.line(rotated_img, (int(s3*scale), int(t3*scale)), (int(s1*scale), int(t1*scale)), (0, 0, 255))
        print (int(s3*scale), int(t3*scale), int(s1*scale), int(t1*scale))
        cv2.imshow("", rotated_img)
        cv2.waitKey(1000)
        cv2.rectangle(rotated_img, (int(xmin*scale),int(ymin*scale)),(int(xmax*scale),int(ymax*scale)), (0, 255, 0), 2)

        cv2.imshow("", rotated_img)
        cv2.waitKey(2000)
    return [xmin, ymin, xmax, ymax], axis_lst

def rotation_by_QRcode(axis_list,rotated_img,angle,scale,flage=True):
    xmin,ymin,xmax,ymax = axis_list

    rotated_img_cp = cv2.resize(rotated_img, (int(rotated_img.shape[1] * scale), int(rotated_img.shape[0] * scale)))
    zbar = rotated_img_cp[int(ymin*scale ):int(ymax*scale ), int(xmin*scale ):int(xmax*scale )]
    if flage:
        cv2.imshow("", zbar)
        cv2.waitKey(2000)
    gray = cv2.cvtColor(zbar, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray, 50, gray.max(), cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=1)
    closed = cv2.erode(closed, None, iterations=2)
    #cv2.imshow("", closed)
    #cv2.waitKey(200)
    if zbar.shape[0] > zbar.shape[1]:
        #shuzhe
        if closed[int(closed.shape[0]/7):int(closed.shape[0]*6/7),int(closed.shape[1]/10):int(closed.shape[1]/2)].mean() > \
                closed[int(closed.shape[0]/7):int(closed.shape[0]*6/7),int(closed.shape[1]/2):int(closed.shape[1]*9/10)].mean():
            #left
            turn = 90
            global_angle = (angle*180/np.pi + 90)/180*np.pi
        elif closed[int(closed.shape[0]/7):int(closed.shape[0]*6/7),int(closed.shape[1]/10):int(closed.shape[1]/2)].mean() <= \
                closed[int(closed.shape[0]/7):int(closed.shape[0]*6/7),int(closed.shape[1]/2):int(closed.shape[1]*9/10)].mean():
            #right
            turn = -90
            global_angle = (angle*180/np.pi - 90)/180*np.pi
    else:
        if closed[int(closed.shape[0]/10):int(closed.shape[0]/2),int(closed.shape[1]/7):int(closed.shape[1]*6/7)].mean() < \
                closed[int(closed.shape[0]/2):int(closed.shape[0]*9/10),int(closed.shape[1]/7):int(closed.shape[1]*6/7)].mean():
            turn = 0
            global_angle = angle
        elif closed[int(closed.shape[0]/10):int(closed.shape[0]/2),int(closed.shape[1]/7):int(closed.shape[1]*6/7)].mean() >=\
                closed[int(closed.shape[0]/2):int(closed.shape[0]*9/10),int(closed.shape[1]/7):int(closed.shape[1]*6/7)].mean():
            turn = 180
            global_angle = (angle*180/np.pi + 180)/180*np.pi
    alter_rotated = rotate(rotated_img, turn)
    axis_lst= rotation_axis(xmin, ymin, xmax, ymax, turn/180*np.pi, rotated_img,
                                alter_rotated, flage)
    qrimg = alter_rotated[axis_lst[1]:axis_lst[3], axis_lst[0]: axis_lst[2]]
    if flage:
        cv2.imshow("", alter_rotated)
        cv2.waitKey(200)

    return alter_rotated, global_angle, qrimg

def other_rotation(image,zuobiao,center):
    (h,w) = image.shape[:2]
    if center is None:
        center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    rotated = cv2.warpAffine(zuobiao, M, (w, h))

def pick_up_content(img, lines, aline, axis_lst,flag,trail):
     #pick_up_content(rotated_img, angle, lines, aline, [ymin[0], ymax[0], xmin[0], xmax[0]])
    trail += 1
    if flag:
        cv2.rectangle(img, (axis_lst[0], axis_lst[1]),(axis_lst[2],axis_lst[3]), (0, 255, 0), 2)
        cv2.imshow("", img)
        cv2.waitKey(2000)
    line1 = 0
    line2 = 0
    line3 = 0
    global_theta = 0
    for line in lines:
        rho, theta = line[0]
        if abs(theta - aline[0][1]) > 0.1 or abs(theta-1.57)>0.25:
            continue

        #print("theta:", theta, "angle:", theta * 180 / np.pi)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        k = float(y1-y2)/(x1-x2)
        q = y1 - k * x1
        yy = int(k * axis_lst[0] + q)
        #y = line[0][0] * np.cos(line[0][1])
        if yy < axis_lst[1] and line1 == 0:
              line1 = min(max(yy,0),img.shape[0])
              if global_theta==0:
                  global_theta += theta
              else:
                  global_theta += theta
                  global_theta /= 2
              if flag:
                  cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
                  cv2.imshow("result", img)
                  cv2.waitKey(500)

        if yy >=  axis_lst[3] and yy < img.shape[0] :
            if 0 < yy - axis_lst[3] < img.shape[0]/5 and line2 == 0:
                line2 = min(max(yy,0),img.shape[0])
                global_theta = theta
                if global_theta == 0:
                    global_theta += theta
                else:
                    global_theta += theta
                    global_theta /= 2
                if flag:
                    cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.imshow("result", img)
                    cv2.waitKey(500)
            elif line3 == 0:
                line3 = min(max(yy,0),img.shape[0]) #[[y1,0],[y2,img.shape[1]]]
                global_theta = theta
                if global_theta == 0:
                    global_theta += theta
                else:
                    global_theta += theta
                    global_theta /= 2
                if flag:
                    cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.imshow("result", img)
                    cv2.waitKey(500)
        if line1 != 0 and line2 != 0 and line3 != 0:
                break
        if flag:
            cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.imshow("result", img)
            cv2.waitKey(200)

    if line1 == 0:
         line1 = int(axis_lst[1]/2)
    if line2 == 0:
         line2 = int(max(int(axis_lst[1]/2+axis_lst[3]), img.shape[0]/2))
    if line3 == 0:
         line3 = int(img.shape[0] - axis_lst[1]/2)
    if (line2-line1 > 30 and line3-line2 > 30) or trail > 4:
        return line1, line2, line3, global_theta
    else:
        lines, aline = alter_HoughLines(img, 10, 2, 100, flag)
        if flag:
            draw_img = draw_line(lines, img, flag, (0, 255, 0))
            cv2.imshow("",draw_img)
            cv2.waitKey(100)
        return pick_up_content(img, lines, aline, axis_lst, flag, trail)

def crop_round_region(axis_lst, img, flage=False, scale=0.3):
    xmin,ymin,xmax,ymax = axis_lst
    xmin_cp = int(xmin*scale)
    ymin_cp = int(ymin*scale)
    xmax_cp = int(xmax*scale)
    ymax_cp = int(ymax*scale)

    (h,w) = img.shape[:2]
    img_cp = cv2.resize(img,(int(w*scale),int(h*scale)))
    if flage:
        cv2.rectangle(img_cp, (xmin_cp, ymin_cp),(xmax_cp,ymax_cp), (0, 255, 0), 2)
        cv2.imshow("",img_cp)
        cv2.waitKey(200)

    receive = img_cp[ymin_cp:ymax_cp, :]
    #cv2.imshow("",receive)
    #cv2.waitKey(2000)
    #crop_distance = max((ymax_cp-ymin_cp),(xmax_cp-xmin_cp))
    crop_distance = img.shape[0]/100
    boxx_ymin = int(max(ymin_cp - crop_distance*1.5, 0))
    boxx_ymax = int(min(ymax_cp + crop_distance*4, img_cp.shape[0]))
    print("box_y:", boxx_ymin, boxx_ymax)
    crop_image = img_cp[int(boxx_ymin):int(boxx_ymax), :]
    if flage:
        cv2.imshow("",crop_image)
        cv2.waitKey(200)
    lines, aline = alter_HoughLines(crop_image, 10, 2, 150, flage)
    #draw_img = draw_line(lines, crop_image, True, (0, 255, 0))
    axis_lst_cp = [xmin_cp, ymin_cp-boxx_ymin, xmax_cp, ymax_cp-boxx_ymin]
    line1, line2, line3, thero = pick_up_content(crop_image, lines, aline, axis_lst_cp, flage,0)
    #draw_img = draw_line(_, draw_img, flage, (0, 0, 255))
    #if thero != 1.57:
    thero_ori = (thero-1.57) * 180 / np.pi
    rotated_ori_img = rotate(img, thero_ori)
    #else:
        #rotated_ori_img = img

    line1_ori = int((line1 + boxx_ymin)/scale)
    line1_end = int((line2 + boxx_ymin)/scale)
    line2_ori = int((line2 + boxx_ymin)/scale)
    line3_ori = int((line3 + boxx_ymin)/scale)
    ori_line_axis = rotation_axis(xmin,line1_ori, xmax,line1_end,(thero-1.57), img, rotated_ori_img, False, scale=0.5)
    region1 = rotated_ori_img[ori_line_axis[1]:ori_line_axis[3], :]
    ori_line_axis = rotation_axis(xmin, line2_ori,xmax, line3_ori, (thero-1.57), img, rotated_ori_img, False, scale=0.5)
    region2 = rotated_ori_img[ori_line_axis[1]:ori_line_axis[3], :]
    ori_box_axis = rotation_axis(xmin, ymin, xmax, ymax, (thero-1.57), img, rotated_ori_img, False, scale=0.5)
    crop_ymin = int(ori_box_axis[1] - ori_line_axis[1])
    crop_ymax = int(ori_box_axis[3] - ori_line_axis[1])
    crop_xmin = int(ori_box_axis[0])
    crop_xmax = int(ori_box_axis[2])
    crop_axis = [crop_xmin, crop_xmax, crop_ymin, crop_ymax]

    if flage:
        cv2.imshow("rotation1:", cv2.resize(region1,(int(region1.shape[1]*scale), int(region1.shape[0]*scale))))
        cv2.waitKey(500)
        crop_axis = (np.array(crop_axis)*scale).astype(int)
        cv2.rectangle(region1, (crop_axis[0], crop_axis[1]), (crop_axis[2], crop_axis[3]), (255, 255, 255), 2)
        cv2.imshow("rotation2:", cv2.resize(region2,(int(region2.shape[1]*scale), int(region2.shape[0]*scale))))
        cv2.waitKey(2000)

    return rotated_ori_img, region1, region2, crop_axis#crop_image

# def hough_line_round_region(img,tflage=False):
#     lines, _ = alter_HoughLines(img,10,flage)
    
def read_label_box(file_dir,label_info_dic,img,class_id,paihang=1):
    height = img.shape[0]
    width = img.shape[1]
    resize_ratio = 0.5
    cimg = cv2.resize(img,(int(resize_ratio*width),int(resize_ratio*height)))
    #cv2.imshow("", cimg)
    #cv2.waitKey(2000)
    detection_boxes = label_info_dic[0].get("detection_boxes")
    detection_scores = label_info_dic[0].get("detection_scores")
    detection_classes = label_info_dic[0].get("detection_classes")
    if paihang == 1:
        index_zabar = np.where(detection_classes == class_id)[0][0]
    elif paihang == 2 and len(np.where(detection_classes == class_id)[0]) > 1:
        index_zabar = np.where(detection_classes == class_id)[0][1]
    else:
        index_zabar = np.where(detection_classes == class_id)[0][0]
    ymin_lst = []
    ymax_lst = []
    xmin_lst = []
    xmax_lst = []
    zbar_lst = []
    #for i in index_zabar:
    ymin = detection_boxes[index_zabar][0]
    xmin = detection_boxes[index_zabar][1]
    ymax = detection_boxes[index_zabar][2]
    xmax = detection_boxes[index_zabar][3]
    #cv2.rectangle(cimg, (int(xmin*width*resize_ratio),int(ymin*resize_ratio*height)), (int(xmax*width*resize_ratio),int(ymax*resize_ratio*height)), (0, 255, 0), 2)
    #cv2.drawContours(cimg, [int(xmin*width*resize_ratio),int(ymin*resize_ratio*height), int(xmax*width*resize_ratio),int(ymax*resize_ratio*height)], -1, 255, thickness=-1)
    zbar = cimg[int(ymin*resize_ratio*height):int(ymax*resize_ratio*height), int(xmin*width*resize_ratio):int(xmax*resize_ratio*width)]
    #cv2.imshow("",zbar)
    #cv2.waitKey(2000)
    ymin_lst.append(ymin*height)
    ymax_lst.append(ymax*height)
    xmin_lst.append(xmin*width)
    xmax_lst.append(xmax*width)
    zbar_lst.append(zbar)

    return xmin_lst, ymin_lst,xmax_lst,ymax_lst, zbar


def read_QRCode_label(file_dir, label_file, class_id=2):
    result_lst = []
    angle_lst = []
    QRimg_lst = []
    if os.path.isdir(file_dir) is False:
        img = cv2.imread(file_dir)
        xmin, ymin, xmax, ymax, zbar = read_label_box(file_dir, label_file, img, 2)
        rotated_img, angle, _, _ = draw_line_rotation_main(file_dir, flage=False)
        axis_lst = rotation_axis(xmin[0], ymin[0], xmax[0], ymax[0], angle, img, rotated_img, flage=False)
        alter_rotated, global_angle, QRimg = rotation_by_QRcode(axis_lst, rotated_img, angle, 0.5,flage=False)
        result_lst.append(alter_rotated)
        angle_lst.append(global_angle)
        QRimg_lst.append(QRimg)
    else:
        label_info = pickle.load(open(label_file, 'rb'))
        for img_name, label_info_dic in label_info.items():
            print("img_name:", img_name)
            # img_name = "00524.jpg"
            img = cv2.imread(os.path.join(file_dir, img_name))
            label_info_dic = label_info.get(img_name)
            xmin, ymin, xmax, ymax, zbar = read_label_box(file_dir, label_info_dic, img, class_id)
            rotated_img, angle, _, _ = draw_line_rotation_main(os.path.join(file_dir, img_name), flage=False)
            axis_lst = rotation_axis(xmin[0], ymin[0], xmax[0], ymax[0], angle, img, rotated_img, flage=False)
            alter_rotated, global_angle, QRimg = rotation_by_QRcode(axis_lst, rotated_img, angle, 0.5,flage=False)
            result_lst.append(alter_rotated)
            angle_lst.append(global_angle)
            QRimg_lst.append(QRimg)
    return result_lst, angle_lst, QRimg_lst

def read_receive_label(file_dir, label_file, class_id):
    label_info = pickle.load(open(label_file, 'rb'))
    all_img_lst = []
    receive_img_lst = []
    sender_img_lst = []
    QR_img_lst = []
    for img_name, label_info_dic in label_info.items():
        print("img_name:", img_name)
        img_name = "00504.jpg"
        label_info_dic = label_info.get(img_name)
        img = cv2.imread(os.path.join(file_dir, img_name))
        alter_rotated_lst, global_angle_lst, QR_img = read_QRCode_label(os.path.join(file_dir, img_name), label_info_dic)

        xmin, ymin, xmax, ymax, zbar = read_label_box(file_dir, label_info_dic, img, class_id,1)
        xmin1, ymin1, xmax1, ymax1, zbar1 = read_label_box(file_dir, label_info_dic, img, class_id,2)
        alter_rotated = alter_rotated_lst[0]
        global_angle = global_angle_lst[0]
        axis_lst = rotation_axis(xmin[0], ymin[0], xmax[0], ymax[0], global_angle, img, alter_rotated,
                                    flage=False)
        axis_lst1 = rotation_axis(xmin1[0], ymin1[0], xmax1[0], ymax1[0], global_angle, img, alter_rotated,
                                    flage=False)
        if axis_lst[1] <= axis_lst1[1]:
            all_img, receive_img, sender_img,crop_axis = crop_round_region(axis_lst, alter_rotated, flage=False)
        else:
            all_img, receive_img, sender_img, crop_axis = crop_round_region(axis_lst1, alter_rotated, flage=False)
        # all_img_lst.append(all_img)
        # receive_img_lst.append(receive_img)
        # sender_img_lst.append(sender_img)
        # QR_img_lst.append(QR_img[0])
    #return all_img_lst, receive_img_lst, sender_img_lst, QR_img_lst

if __name__ == "__main__":
    #read_QRCode_label("/home/gytang/yuantong/huitong20171108","/home/gytang/yuantong/huitong/result.pkl",2)
    read_receive_label("/home/gytang/yuantong/huitong20171108","/home/gytang/yuantong/huitong/result.pkl", 1)
