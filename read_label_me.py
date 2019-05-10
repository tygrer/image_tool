import os
import cv2
import numpy as np
import xml.dom.minidom as DOM
import xml.etree.cElementTree as etree
import xml.etree.ElementTree as ET




def rotate(image,angle,center = None,scale=1.0):
    (h,w) = image.shape[:2]
    if center is None:
        center = (w/2, h/2)

    M = cv2.getRotationMatrix2D((int(w/2),int(h/2)), angle, scale)
    #print("s1:", s, "t1:", t)
    #print("w:", w, "h:", h)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    # print(M[0,0],M[0,1])
    # print(np.sin(-angle),np.cos(-angle))
    nw = int((h*sin)+(w*cos))
    nh = int((h*cos)+(w*sin))
    M[0,2] += (nw/2) - w/2
    M[1,2] += (nh/2) - h/2
    rotated = cv2.warpAffine(image, M, (nw, nh))
    #rotated = cv2.warpAffine(image, M, (w, h))
    # print("s1:", rotated.shape[1], "t1:", rotated.shape[0])
    # a = cv2.resize(rotated, (int(nw/4), int(nh/4)))
    #cv2.imshow("",a)
    #cv2.waitKey(2000)
    return rotated


def crop_image_and_rotate_fc(img, xmin,ymin,xmax,ymax,thero_ori):
    flag = True
    (h, w) = img.shape[:2]
    # if flag:
    #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #     cv2.imshow("", img)
    #     cv2.waitKey(200)

    # draw_img = draw_line(lines, crop_image, True, (0, 255, 0))
    # draw_img = draw_line(_, draw_img, flage, (0, 0, 255))
    # if thero != 1.57:
    # thero_ori = (thero - 1.57) * 180 / np.pi
    crop_img = img[ymin:ymax,xmin:xmax]
    rotated_ori_img = rotate(crop_img, thero_ori)

    # if flag:
    #     cv2.imshow("rotation1:", rotated_ori_img)
    #     cv2.waitKey(500)

    return rotated_ori_img  # crop_image

def arrange_hc_label_me():
    jpg_dir = '/home/gytang/project/traintickets/image/'
    xml_dir = '/home/gytang/project/traintickets/label/'
    zaiqing_xml_dir = '/home/gytang/project/zq_traintickets/'

    txt_label = '/home/gytang/project/traintickets/traintickets_info.txt'
    ft = open(txt_label, 'w')
    for info in os.listdir(xml_dir):
        with open(os.path.join(xml_dir, info), 'r') as f:
            # class_rst_dict = defaultdict(list)
            # # we fill the data path not label path in the class_rst_dict. It is used for separation
            # filled_value = filled_file_name if filled_file_name is not None else file_path
            # with open(file_path, 'r') as f:
            fpimage = cv2.imread(os.path.join(jpg_dir, os.path.splitext(info)[0] + '.jpg'))

            xml_tree = ET.parse(f)
            root = xml_tree.getroot()
            file_name = root.find('filename').text
            class_rst_lst = {}
            for_direction = 0
            obj_lst = root.findall("object")
            angel = None
            for obj_id, obj in enumerate(obj_lst):
                cls = obj.find('name').text
                spls = cls.split('-')
                if len(spls) < 3:
                    if cls == 'hcp-text':
                        continue
                    else:
                        print("exception label name: ", cls,info)
                elif len(spls) == 3:
                    label = spls[-1]
                    class_rst_lst['name'] = label
                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    if label in ['up', 'down', 'left', 'right']:
                        for_direction += 1
                        try:
                            with open(os.path.join(zaiqing_xml_dir, info), 'r') as directf:
                                xml_tree_directf = ET.parse(directf)
                                directroot = xml_tree_directf.getroot()
                                direct_obj_lst = directroot.findall('object')
                                direct_for_direction = 0
                                for direct_obj_id, direct_obj in  enumerate(direct_obj_lst):
                                    if direct_obj.find('name').text.split('-')[-1] in ['up', 'down', 'left', 'right']:
                                        direct_for_direction += 1
                                        if for_direction == direct_for_direction:
                                            direct_bndbox = direct_obj.find('bndbox')
                                            # if str(xmin) == direct_bndbox.find('xmin').text \
                                            # and str(ymin) == direct_bndbox.find('ymin').text \
                                            # and str(xmax) == direct_bndbox.find('xmax').text \
                                            # and str(ymax) == direct_bndbox.find('ymax').text:
                                            direct =  direct_obj.find('name').text.split('-')[-1]
                                            if direct == 'right':
                                                angel = 90
                                            elif direct == 'left':
                                                angel = -90
                                            elif direct == 'down':
                                                angel = 180
                                            else:
                                                angel = 0
                                            co_img = crop_image_and_rotate_fc(fpimage, xmin, ymin, xmax, ymax, angel)
                                                # for next_obj_lst in enumerate(obj_lst[obj_id:]):
                                                #     next_spls = next_obj_lst.find('name').text.split('-')
                                                #     if len(next_spls) == 3 and next_spls[-1] not in ['up', 'down', 'left', 'right']:
                                                #         next_bndbox = next_obj_lst.find("bndbox").text
                                                #         xmin_next = int(next_bndbox.find('xmin').text)
                                                #         ymin_next = int(next_bndbox.find('ymin').text)
                                                #         xmax_next = int(next_bndbox.find('xmax').text)
                                                #         ymax_next = int(next_bndbox.find('ymax').text)
                                                #         co_img = crop_image_and_rotate_fc(fpimage, xmin_next, ymin_next, xmax_next, ymax_next, rotate)
                                            # else:
                                            #     print(xmin,ymin,xmax,ymax," | ",direct_bndbox.find('xmin').text,
                                            #           direct_bndbox.find('ymin').text,
                                            #           direct_bndbox.find('xmax').text,
                                            #           direct_bndbox.find('ymax').text)

                        except:
                            print("zq has no this xml file.", info)
                    else:
                        if angel is not None:
                            co_img = crop_image_and_rotate_fc(fpimage, xmin, ymin, xmax, ymax, angel)
                            cv2.imwrite("/home/gytang/project/traintickets/crop_image/"+os.path.splitext(info)[0]+'_'+str(obj_id)+'.jpg', co_img)
                            ft.write(os.path.splitext(info)[0] + '_'+ str(obj_id)+'.jpg' + ':' + label + '\n')
                        else:
                            print("exception label, the angel is None,",info)
                else:
                    print("exception label name: ", cls, info)
    ft.close()

def arrange_jp_label_me():
    jpg_dir = '/home/gytang/project/dataset/yg_railway_airplane/air/jpg/'
    xml_dir = '/home/gytang/project/dataset/yg_railway_airplane/air/label/'
    #zaiqing_xml_dir = '/home/gytang/project/zq_traintickets/'

    txt_label = '/home/gytang/project/planetickets/planetickets_info.txt'
    ft = open(txt_label, 'w')
    for info in os.listdir(xml_dir):
        with open(os.path.join(xml_dir, info), 'r') as f:
            # class_rst_dict = defaultdict(list)
            # # we fill the data path not label path in the class_rst_dict. It is used for separation
            # filled_value = filled_file_name if filled_file_name is not None else file_path
            # with open(file_path, 'r') as f:
            fpimage = cv2.imread(os.path.join(jpg_dir, os.path.splitext(info)[0] + '.jpg'))

            xml_tree = ET.parse(f)
            root = xml_tree.getroot()
            class_rst_lst = {}
            for_direction = 0
            obj_lst = root.findall("object")
            angel = None
            for obj_id, obj in enumerate(obj_lst):
                cls = obj.find('name').text
                spls = cls.split('-')
                if len(spls) < 3:
                    if cls == 'jp-text':
                        continue
                    else:
                        label = spls[-1]
                        class_rst_lst['name'] = label
                        bndbox = obj.find("bndbox")
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        if label in ['up', 'down', 'left', 'right']:
                            for_direction += 1
                            if label == 'right':
                                angel = -90
                            elif label == 'left':
                                angel = 90
                            elif label == 'down':
                                angel = 180
                            else:
                                angel = 0
                            co_img = crop_image_and_rotate_fc(fpimage, xmin, ymin, xmax, ymax,
                                                              angel)
                elif len(spls) >= 3:
                    label = ""
                    for lab_i in spls[2:]:
                        label += '-'+lab_i
                    if label[0]=='-':
                        label=label[1:]
                    class_rst_lst['name'] = label
                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    if angel is not None:
                        if xmin>=xmax or ymin>=ymax:
                            print("crop content failed ",cls,info)
                        else:
                            co_img = crop_image_and_rotate_fc(fpimage, xmin, ymin, xmax, ymax, angel)

                        cv2.imwrite("/home/gytang/project/dataset/yg_railway_airplane/air/crop_image/" + os.path.splitext(info)[
                            0] + '_' + str(obj_id) + '.jpg', co_img)
                        ft.write(os.path.splitext(info)[0] + '_' + str(obj_id) + '.jpg' + ':' + label + '\n')
                    else:
                        print("exception label, the angel is None,", info)

                else:
                    print("exception label name: ", cls, info)
    ft.close()

def arrange_jp_train_label_me():
    jpg_dir = '/home/gytang/project/dataset/yg_railway_airplane/air/jpg/'
    xml_dir = '/home/gytang/project/dataset/yg_railway_airplane/air/label/'
    #zaiqing_xml_dir = '/home/gytang/project/zq_traintickets/'
    fr = open('/home/gytang/project/dataset/yg_railway_airplane/air/train.txt', 'r')
    lines= fr.readlines()
    txt_label = '/home/gytang/project/dataset/yg_railway_airplane/air/planetickets_info.txt'
    ft = open(txt_label, 'w')
    for info in lines:
    #for info in os.listdir(xml_dir):
        info = info[:-3]
        with open(os.path.join(xml_dir, info+'.xml'), 'r') as f:
            # class_rst_dict = defaultdict(list)
            # # we fill the data path not label path in the class_rst_dict. It is used for separation
            # filled_value = filled_file_name if filled_file_name is not None else file_path
            # with open(file_path, 'r') as f:
            fpimage = cv2.imread(os.path.join(jpg_dir, info + '.jpg'))

            xml_tree = ET.parse(f)
            root = xml_tree.getroot()
            class_rst_lst = {}
            for_direction = 0
            obj_lst = root.findall("object")
            angel = None
            for obj_id, obj in enumerate(obj_lst):
                cls = obj.find('name').text
                spls = cls.split('-')
                if len(spls) < 3:
                    if 'jp-text' in cls:
                        continue
                    else:
                        label = spls[-1]
                        class_rst_lst['name'] = label
                        bndbox = obj.find("bndbox")
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        if label in ['up', 'down', 'left', 'right']:
                            for_direction += 1
                            if label == 'right':
                                angel = 90
                            elif label == 'left':
                                angel = -90
                            elif label == 'down':
                                angel = 180
                            else:
                                angel = 0
                            co_img = crop_image_and_rotate_fc(fpimage, xmin, ymin, xmax, ymax,
                                                              angel)
            for obj_id, obj in enumerate(obj_lst):
                cls = obj.find('name').text
                spls = cls.split('-')
                if len(spls) >= 3:
                    label = ""
                    for lab_i in spls[2:]:
                        label += '-'+lab_i
                    if label[0]=='-':
                        label=label[1:]
                    class_rst_lst['name'] = label
                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    if angel is not None:
                        if xmin>=xmax or ymin>=ymax:
                            print("crop content failed ", cls, info)
                        else:
                            co_img = crop_image_and_rotate_fc(fpimage, xmin, ymin, xmax, ymax, angel)

                        cv2.imwrite("/home/gytang/project/dataset/yg_railway_airplane/air/crop_image/" + info + '_' + str(obj_id) + '.jpg', co_img)
                        ft.write(info + '_' + str(obj_id) + '.jpg' + ':' + label + '\n')
                    else:
                        print("exception label, the angel is None,", info)

    ft.close()

if __name__ == "__main__":
    #arrange_hc_label_me()
    arrange_jp_train_label_me()

