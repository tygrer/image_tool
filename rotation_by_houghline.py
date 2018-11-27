import cv2
import numpy as np
import os
import math

def translate(image,x,y):
    M = np.float32([[1,0,x],[0,1,y]])
    shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
    return shifted

def rotate(image,angle,center = None,scale=1.0):
    (h,w) = image.shape[:2]
    if center is None:
        center = (w/2, h/2)


    M = cv2.getRotationMatrix2D((int(w/2),int(h/2)), angle, scale)
    #print("s1:", s, "t1:", t)
    print("w:", w, "h:", h)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    print(M[0,0],M[0,1])
    print(np.sin(-angle),np.cos(-angle))
    nw = int((h*sin)+(w*cos))
    nh = int((h*cos)+(w*sin))
    M[0,2] += (nw/2) - w/2
    M[1,2] += (nh/2) - h/2
    rotated = cv2.warpAffine(image, M, (nw, nh))
    #rotated = cv2.warpAffine(image, M, (w, h))
    print("s1:", rotated.shape[1], "t1:", rotated.shape[0])
    a = cv2.resize(rotated, (int(nw/4), int(nh/4)))
    #cv2.imshow("",a)
    #cv2.waitKey(2000)
    return rotated

def max_mid(lines,threhold):
    easyline = lines.reshape(lines.shape[0],lines.shape[-1])
    index_sort = np.argsort(easyline, axis=0)
    sort_list = np.zeros((lines.shape[0],2))
    sum_max_count_angle = easyline[index_sort[0, 1],1]
    sum_scount_angle = easyline[index_sort[0, 1],1]
    max_count_rho = easyline[index_sort[0, 1],0]
    max_count = 1
    cur_count = 1
    for i in range(1,index_sort.shape[0]):
        sort_list[i] = easyline[index_sort[i, 1]]
        if (sort_list[i,1] - easyline[index_sort[i-1, 1],1]) < threhold:
            cur_count += 1
            sum_scount_angle += easyline[index_sort[i, 1],1]
        else:
            sum_scount_angle = easyline[index_sort[i, 1],1]
            cur_count = 1
        if max_count < cur_count:
            max_count = cur_count
            sum_max_count_angle = sum_scount_angle
            max_count_rho = easyline[index_sort[i, 1],0]
    max_count_angle = sum_max_count_angle/max_count
    return max_count_rho, max_count_angle

def find_other_value_in_ndarray(ndarr,value):
    for i in range(ndarr.shape[0]):
        if ndarr[i,0,1] == value:
            ret_value = ndarr[i,0,0]
    return ret_value

def planB_fake(image,hist):
    print("in the planB")
    image = cv2.medianBlur(image, 7)
    # image = cv2.GaussianBlur(image,(3,3),0)
    kernel_sharpen_2 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]]) / 8.0
    image = cv2.filter2D(image, -1, kernel_sharpen_2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray, np.argmax(hist), gray.max(), cv2.THRESH_BINARY)
    #cv2.imshow("", thresh)
    #cv2.waitKey(1000)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    #cv2.imshow("", edges)
    #cv2.waitKey(1000)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    lines_lst = []
    for i, _ in enumerate(lines):
        if lines[i][0][1] != np.float32(0.7853982):
            lines_lst.append(lines[i])
    lines = np.array(lines_lst)
    if lines is not None:
        rho = lines[0,0,0]
        theta = lines[0, 0,1]
    return rho, theta

def planB_draw_line(img,result,lines):
    for line in lines:
        rho, theta = line[0]
        print("theta:", theta, "angle:", theta * 180 / np.pi)
        a = np.cos(theta)
        b = np.sin(theta)
        if (theta < (np.pi / 4)) or (theta > (3. * np.pi / 4)):
            pt1 = (int(rho / np.cos(theta)), 0)
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            cv2.line(result, pt1, pt2, (255, 255, 0))
            print("angle1:", theta / np.pi * 180)
            img = rotate(img, theta / np.pi * 180 - 90)
        else:
            pt1 = (0, int(rho / np.sin(theta)))
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            cv2.line(result, pt1, pt2, (255, 255, 0), 1)
            print("angle2:", theta / np.pi * 180)
            img = rotate(img, theta / np.pi * 180 - 90)

def draw_line(lines,img,flag=False, color=(0,0,255)):
    result = img.copy()
    for line in lines:
        rho, theta = line[0]
        #print("theta:", theta, "angle:", theta * 180 / np.pi)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if flag:
            cv2.line(result, (x1, y1), (x2, y2), color, 2)
            cv2.imshow("result", result)
            cv2.waitKey(200)
    return result

def alter_HoughLines(image,high_thre, low_thre, init_threshold, flage=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])

    (_, thresh) = cv2.threshold(gray, init_threshold, gray.max(), cv2.THRESH_BINARY)
    if flage:
        cv2.imshow("", thresh)
        cv2.waitKey(1000)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    if flage:
        cv2.imshow("", edges)
        cv2.waitKey(1000)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    theta = None
    trail = 0

    while theta is None:
        trail += 1
        if trail > 10:
            print("in the planD")
            if lines is not None:
                theta = lines[0, 0, 1]
                rho = lines[0, 0, 0]
                lines = np.array([[rho, theta]])
                aline = lines
                break
            else:
                theta = 0
                rho = 20
                lines = np.array([[rho, theta]])
                aline = lines
                break
        if lines is None:
            print("in the planB")
            (_, thresh) = cv2.threshold(gray, np.argmax(hist), gray.max(), cv2.THRESH_BINARY)
            if flage:
                cv2.imshow("", thresh)
                cv2.waitKey(1000)
            edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
            if lines is None:
                thre = 50
                while lines is None or thre < 255:
                    print("in the planC")
                    thre = trail * 10
                    trail += 1
                    if trail > 10:
                        break
                    (_, thresh) = cv2.threshold(gray, thre, gray.max(), cv2.THRESH_BINARY)
                    if flage:
                        cv2.imshow("", thresh)
                        cv2.waitKey(1000)
                    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
                    if lines is not None:
                        break
            if lines is not None and len(lines) < 35:
                lines_lst = []
                for i, _ in enumerate(lines):
                    if lines[i][0][1] != np.float32(0.7853982) and lines[i][0][1] != np.float32(2.3561945):
                        lines_lst.append(lines[i])
                lines = np.array(lines_lst)
                rho, theta = max_mid(lines, 0.1)
                aline = np.array([[rho, theta]])
            else:
                continue
        elif len(lines) < high_thre and len(lines) > low_thre :
            print("in the planA")
            lines_lst = []
            for i, _ in enumerate(lines):
                if lines[i][0][1] != np.float32(0.7853982) and lines[i][0][1] != np.float32(2.3561945):
                    lines_lst.append(lines[i])
            lines = np.array(lines_lst)
            rho, theta = max_mid(lines, 0.1)
            aline = np.array([[rho, theta]])
        else:
            print("in the planE")
            thre = 85
            while len(lines) > high_thre or thre < 255:
                print("in the planF")
                thre = trail * 20
                trail += 1
                if trail > 10:
                    break
                (_, thresh) = cv2.threshold(gray, thre, gray.max(), cv2.THRESH_BINARY)
                if flage:
                    cv2.imshow("", thresh)
                    cv2.waitKey(1000)
                edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
                if lines is not None:
                    if len(lines) > 35:
                        continue
                    else:
                        break
                else:
                    break
            if lines is not None:
                lines_lst = []
                for i, _ in enumerate(lines):
                    if lines[i][0][1] != np.float32(0.7853982) and lines[i][0][1] != np.float32(2.3561945):
                        lines_lst.append(lines[i])
                lines = np.array(lines_lst)
                rho, theta = max_mid(lines, 0.1)
                aline = np.array([[rho, theta]])
            else:
                continue
    return lines, aline


def draw_line_rotation_main(filedir,flage=False):
    if os.path.isdir(filedir):
        file_lst = os.listdir(filedir)
        result = []
        angle = []
        for i in range(20,len(file_lst)):
            print("i:",i)
            if os.path.splitext(file_lst[i])[-1] != '.jpg':
                continue
            image = cv2.imread(os.path.join(filedir, file_lst[i]))

            img = cv2.resize(image, (int(image.shape[1]*0.5), int(image.shape[0]*0.5)))
            lines, aline = alter_HoughLines(img, 35,2,150, flage)
            if flage:
                draw_img = draw_line(lines, img, flage, (0, 255, 0))
                draw_img = draw_line(aline, draw_img, flage, (0, 0, 255))
                cv2.imshow("rotation:", draw_img)
                cv2.waitKey(200)
            result_one = rotate(image, aline[0,1]* 180 / np.pi-90 )
            result.append(result_one)
            if (aline[0, 0, 1] < (np.pi / 4)) or (aline[0, 0, 1] > (3. * np.pi / 4)):
                angle = (aline[ 0, 1] * 180 / np.pi + 90) * np.pi / 180
            else:
                angle = (aline[ 0, 1] * 180 / np.pi - 90) * np.pi / 180
            angle.append(angle)
    else:
        if os.path.splitext(filedir)[-1] != '.jpg':
            print("This is not a jpg format image.")
            return
        image = cv2.imread(os.path.join(filedir))
        img = cv2.resize(image, (512,512))
        lines, aline = alter_HoughLines(img,35,2,150, flage)
        if flage:
            draw_img = draw_line(lines, img, flage, (0, 255, 0))
            draw_img = draw_line(aline, draw_img, flage, (0, 0, 255))
            cv2.imshow("rotation:", draw_img)
            cv2.waitKey(200)

        if (aline[0, 1] < (np.pi / 4)) or (aline[0, 1] > (3. * np.pi / 4)):
            angle = (aline[0, 1] * 180 / np.pi + 90) * np.pi / 180
            result = rotate(image, aline[0, 1] * 180 / np.pi + 90)
            #result = rotate(image, angle)
            print(aline[0, 1], "angle is the normal:", angle/np.pi * 180)
        else:
            angle = (aline[0, 1] * 180 / np.pi - 90) * np.pi / 180
            result = rotate(image, aline[0, 1] * 180 / np.pi - 90)
            #result = rotate(image,angle)
            print(aline[0, 1], "angle is not normal:", angle / np.pi * 180)
    return result, angle, lines, aline

if __name__ == "__main__":
    rota_image = draw_line_rotation_main("/home/gytang/yuantong/huitong20171108/")
