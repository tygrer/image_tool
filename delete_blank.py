import os
import cv2
import numpy as np
img_dir = "/home/gytang/project/dataset/2019.1.30/crop_train/"
save_dir = "/home/gytang/project/dataset/2019.1.30/train_n/"
# label_dir = "/home/gytang/project/dataset/2019.1.30/txt/"
#dir_b = "/media/gytang/My Passport3/RESIDE/synthetic/original"
file_list = os.listdir(img_dir)
file_list.sort()
# b_list = os.listdir(dir_b)
# b_list.sort()
flag = 0
# write_txt = open('./defp_2.19_val.txt','w')
for _,imga in enumerate(file_list):
    # for _,imga in enumerate(os.listdir(os.path.join(dir,file))):
        #for b_idx, imgb in enumerate(b_list):
        content_a = cv2.imread(os.path.join(img_dir,imga))
        start_row = None
        end_row = None
        crop_image=None
        # cv2.imshow("", content_a)
        # cv2.waitKey(1000)
        for i in range(content_a.shape[0]-1):
            # for j in range(content_a.shape[1]):
            #print(np.mean(content_a[i, :].flatten()),np.mean(content_a[i + 1, :].flatten()))
            if np.mean(content_a[i, :].flatten()) >=253 and np.mean(content_a[i + 1, :].flatten()) < 253:
                start_row = i+2

            elif np.mean(content_a[i, :].flatten())<253 and np.mean(content_a[i + 1, :].flatten()) >=253:
                end_row = i
                try:
                    crop_image=content_a[start_row:start_row+32,:,:]
                    crop_image = np.uint8(np.concatenate((crop_image, np.ones((crop_image.shape[0], 1, 3)) * 255), axis=1))

                    # cv2.imshow("", crop_image)
                    # cv2.waitKey(1000)
                except:
                    pass
                    print(os.path.join(img_dir,imga))
                start_row = None
                end_row = None
                break
        start_line = None
        end_line = None
        if crop_image is None:
            if content_a.shape[0]<34:
                crop_image  = content_a
                cv2.imwrite(os.path.join(save_dir, imga), crop_image)
                continue
            else:
                crop_image = content_a[int((content_a.shape[0]-34)/2):int(content_a.shape[0]-(content_a.shape[0]-34)/2),:,:]

        for j in range(content_a.shape[1]):
            final_crop_image=None
            # print(np.mean(crop_image[:,j,:].flatten()))
            if np.mean(crop_image[:,j,:].flatten())>253 and np.mean(crop_image[:,j+1,:].flatten())<253:
                start_line = j+2
            elif start_line is not None and np.mean(crop_image[:,j,:].flatten())<253 and np.mean(crop_image[:,j+1,:].flatten())>253:
                end_line = j
                if  end_line-start_line>crop_image.shape[1]*0.8:
                    final_crop_image = crop_image[:, start_line:end_line, :]
                    cv2.imwrite(os.path.join(save_dir,imga),final_crop_image)
                    # cv2.imshow("", final_crop_image)
                    # cv2.waitKey(1000)
                    start_line = None
                    end_line = None
                    break
        if final_crop_image is None:
            # cv2.imshow("",crop_image[:,5:,:])
            # cv2.waitKey(1000)
            print("***",imga)
            cv2.imwrite(os.path.join(save_dir, imga), crop_image[:,5:,:])






