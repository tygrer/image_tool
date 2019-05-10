import random, cv2, pickle,os
import numpy as np
img_label_all = []
img_name_all = []
root_dir = '/home/gytang/project/dataset/yg_bak_0722/'
def detection_type(vattype):
    global ctpn_weights, ocr_weights
    if vattype == 'defp' or vattype == 'all':
        input_file = os.path.join(root_dir, "defp_train.pkl")#defp_20190104.pkl
        image_dir = os.path.join(root_dir, "de")#de20190104
        val_data = os.path.join(root_dir, "defp_train.pkl")#defp_20190104.pkl
        old_data = os.path.join(root_dir, "defp_train.pkl")


    elif vattype == 'vat':
        input_file = os.path.join(root_dir, "invoice_val.pkl")
        image_dir = os.path.join(root_dir, "images")
        val_data = os.path.join(root_dir, "invoice_val.pkl")
        old_data = os.path.join(root_dir, "invoice_val.pkl")

    elif vattype == 'digital':
        input_file = os.path.join(root_dir, "digital_train.pkl")
        image_dir = os.path.join(root_dir, "dzp")
        val_data = os.path.join(root_dir, "digital_train.pkl")
        old_data = os.path.join(root_dir, "digital_train.pkl")

    elif vattype == 'rail':
        input_file = os.path.join(root_dir, 'rail_train.pkl')
        image_dir = os.path.join(root_dir, 'rail')
        val_data = os.path.join(root_dir, 'rail_train.pkl')
        old_data = os.path.join(root_dir, 'rail_train.pkl')
        ctpn_weights = 'checkpoint/graph_ctpn_rail'
        ocr_weights = 'checkpoint/graph_ocr_rail_0308_1'

    elif vattype == 'roll':
        input_file = os.path.join(root_dir, 'roll_ticket.pkl')
        image_dir = os.path.join(root_dir, 'roll')
        val_data = os.path.join(root_dir, 'roll_ticket.pkl')
        old_data = os.path.join(root_dir, 'roll_ticket.pkl')
        ctpn_weights = 'checkpoint/graph_ctpn_roll_0327'
        ocr_weights = 'checkpoint/ctc_graph_jp_0411'

    elif vattype == 'plane':
        input_file = os.path.join(root_dir, 'air_val.pkl')
        image_dir = os.path.join(root_dir, 'plane_padding')
        val_data = os.path.join(root_dir, 'air_val.pkl')
        old_data = os.path.join(root_dir, 'air_val.pkl')
        ctpn_weights = 'checkpoint/graph_ctpn_air_0318'
        ocr_weights = 'checkpoint/graph_ocr_air_0318'

    elif vattype == 'mix':
        input_file = os.path.join(root_dir, 'mix_val.pkl')
        image_dir = os.path.join(root_dir, 'mix')
        val_data = os.path.join(root_dir, 'mix_val.pkl')
        old_data = os.path.join(root_dir, 'mix_val.pkl')
        ctpn_weights = 'checkpoint/graph_ctpn_air_0318'
        ocr_weights = 'checkpoint/graph_ocr_air_0318'

    return input_file, image_dir, val_data, old_data

for j in range(100):
    glob_height = 4000
    glob_width = 2000
    fin_img = (np.ones((glob_height, glob_width, 3)) * 255).astype(np.uint8)
    point_width = 10
    point_height = 10
    point_end = 10
    point_end_w = 10
    img_label_lst = []
    img_lst = []
    train_height = []
    train_width = []
    for i in range(8):
        vattype_lst = ['rail']
        vattype = vattype_lst[0]
        input_file, image_dir, val_data, old_data = detection_type(vattype)
        fpkl = open(input_file, 'rb')
        img_name, img_label = pickle.load(fpkl)
        save_no = random.randint(0, len(img_name) - 1)
        img = cv2.imread(os.path.join(image_dir, img_name[save_no]))
        img_lst.append((vattype, img, img_label[save_no]))
        # img_lst.append(img)
        height, width = img.shape[:2]
        if vattype == 'rail':
            train_height.append(height)
            train_width.append(width)
    if train_height != []:
        max_width = max(train_width)
        max_height = max(train_height)

    for imgg in img_lst:
        img = imgg[1]
        vattype = imgg[0]
        if imgg[0] == 'rail':
            img = cv2.resize(img, (max_width,max_height))
        height, width = img.shape[:2]
        angle = random.randint(-5, 5)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))
        height, width = img.shape[:2]
        if (point_width + width) < glob_width:
            try:
                fin_img[point_height:point_height + height, point_width:point_width + width, :] = img
                point_end_w = max(point_end_w, point_width + width + 10)
                point_end = max(point_end, point_height + height + 10)
                img_label_lst.append((vattype, imgg[2]))
                point_width = point_width + width + 10
            except:
                print(height, width, point_height, point_width)
        else:

            if (point_end + height) < glob_height:
                #try:
                point_width = 10
                point_height = point_end
                fin_img[point_height:point_height + height, point_width:point_width + width, :] = img
                #point_end = point_end + height + 10
                point_width = point_width + width + 10
                img_label_lst.append((vattype, imgg[2]))
                # except:
                #     print(height, width, point_height, point_width)
            else:
                break

    if point_end < glob_height:
        finimg = fin_img[:point_end, :, :]
    if point_end_w < glob_width:
        finimg = finimg[:, :point_end_w, :]
    final_height, final_width = finimg.shape[:2]
    result_img = (np.ones((1300, 2500, 3)) * 255).astype(np.uint8)
    sta_h = max(int((1300 - final_height) / 3),0)
    sta_w = max(int((2500 - final_width) / 2),0)
    if final_height > 1300 or final_width > 2500:
        result_img = cv2.resize(result_img, (max(2500, final_width), max(1300, final_height+200)))
        sta_h = sta_h+100
    result_img[sta_h:sta_h + final_height, sta_w:sta_w + final_width, :] = finimg
    jpgname = "mix" + str(j) + '.jpg'
    img_name_all.append(jpgname)
    # cv2.imwrite(os.path.join('/home/gytang/project/dataset/yg_bak_0722/mix_train', jpgname), result_img)
    img_label_all.append(img_label_lst)
    # finimg = cv2.resize(finimg, (512, 512))
    # cv2.imshow("", finimg)
    # cv2.waitKey(1000)

# output = open('/home/gytang/project/dataset/yg_bak_0722/mix_train_val.pkl', 'wb')
# pickle.dump((img_name_all, img_label_all), output)