import os
import cv2
import numpy as np
img_dir = "/home/gytang/project/dataset/2019.1.30/jpg/"
label_dir = "/home/gytang/project/dataset/2019.1.30/txt/"
#dir_b = "/media/gytang/My Passport3/RESIDE/synthetic/original"
file_list = os.listdir(img_dir)
file_list.sort()
# b_list = os.listdir(dir_b)
# b_list.sort()
flag = 0
write_txt = open('./defp_2.19_train_crop.txt','w')
for _,imga in enumerate(file_list[:120]):
    # for _,imga in enumerate(os.listdir(os.path.join(dir,file))):
        #for b_idx, imgb in enumerate(b_list):
        content_a = cv2.imread(os.path.join(img_dir,imga))
        txt_a = open(os.path.join(label_dir,os.path.splitext(imga)[0]+'.txt'),'r')
        for line in range(content_a.shape[1])[20:]:
            if min((content_a[:,line,0]).flatten())==255:
                for line_contine in range(content_a.shape[1])[line:]:
                    if min(content_a[:,line_contine,0].flatten())<255:
                        break
                #---find the image start crop--
                startrow = None
                endrow = None

                for row in range(content_a.shape[0])[:-1]:
                    con = cv2.cvtColor(content_a,cv2.COLOR_BGR2GRAY)
                    if min(con[row,line_contine:].flatten())==255 and min(con[row+1,line_contine:].flatten())<255:
                        startrow = row+1
                    elif min(con[row,line_contine:].flatten())<255 and min(con[row+1,line_contine:].flatten())==255:
                        endrow  = row
                        crop_content = content_a[startrow:endrow,line_contine-1:,:]
                        crop_lst = []
                        con_crop = cv2.cvtColor(crop_content, cv2.COLOR_BGR2GRAY)
                        startrow = None
                        endrow = None
                        for line_interva in range(crop_content.shape[1])[:-1]:
                            if endrow is None and min(con_crop[:,line_interva].flatten())==255 and  min(con_crop[:,line_interva+1].flatten())<255:
                                startrow = line_interva+1
                                endrow=None
                            elif startrow is not None and min(con_crop[:,line_interva].flatten())<255 and min(con_crop[:,line_interva+1].flatten())==255:
                                endrow = line_interva
                                if endrow-startrow>30:
                                    crop_img = crop_content[:,startrow:endrow,:]
                                    crop_lst.append(crop_img)
                                    # cv2.imshow("",crop_img)
                                    # cv2.waitKey(1000)
                                    startrow = None
                                    endrow = None
                        line_txt = txt_a.readline()
                        try:
                            txt_lst = line_txt.split(':')[1].split(' ')
                        except:
                            print(line_txt)
                        if "*" in txt_lst:
                            continue
                        if len(crop_lst)+1 == len(txt_lst):

                            for si,st in enumerate(txt_lst[1:]):
                                # cv2.imshow("",crop_lst[si])
                                # cv2.waitKey(700)
                                if line_txt.split(':')[0].split('.')[0] =="":
                                    print(line_txt)
                                if st == "":
                                    continue

                                if st =="\n":
                                    continue
                                cv2.imwrite("/home/gytang/project/dataset/2019.1.30/crop_train2/"+
                                            line_txt.split(':')[0].split('.')[0]+'_'+str(flag)+'.jpg',crop_lst[si])

                                if '\n' not in st:
                                    write_txt.write(
                                        '/iqubicdata/workspace/tanggy/project/rcnn/dataset/2019.1.30/train/'+line_txt.split(':')[0].split('.')[0] + '_' + str(flag) + '.jpg' + ':'+st+'\n')
                                else:
                                    write_txt.write('/iqubicdata/workspace/tanggy/project/rcnn/dataset/2019.1.30/train/'+
                                        line_txt.split(':')[0].split('.')[0] + '_' + str(
                                            flag) + '.jpg' + ':' + st)

                                flag += 1
                break
        txt_a.close()
write_txt.close()
        # content_b = cv2.imread(os.path.join(dir_b, b_list[a_idx]))
        # content_b = cv2.resize(content_b, (128, 128))H\
        # content = np.concatenate((content_a,content_b),axis=1)
        # if not os.path.exists(os.path.join("/media/gytang/My Passport3/exp_res/car-detection/car-resize",file)):
        #     os.mkdir(os.path.join("/media/gytang/My Passport3/exp_res/car-detection/car-resize",file))
        # cv2.imwrite(os.path.join("/media/gytang/My Passport3/exp_res/car-detection/car-resize",file,imga+".jpg",),content_a)
