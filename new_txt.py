# alter width and height only fen with cropped
import os, glob, shutil, cv2
import xml.dom.minidom as DOM
import xml.etree.cElementTree as etree
import xml.etree.ElementTree as ET


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    siz = root.findall('size')[0]
    width = siz.find('./width').text
    height = siz.find('./height').text
    obj = root.findall('object')[0]
    return width, height


xml_template_file = "/home/gytang/fen_no.xml"
xml_template_content = open(xml_template_file).readlines()
xml_template_content = ''.join(xml_template_content)
width_template, height_template = parse_xml(xml_template_file)
# print(label_template, xmin_template, ymin_template, xmax_template, ymax_template, width_template, height_template)
print(xml_template_content)
img_dir = "/home/gytang/huangwen/data/zheyun/lowamt/fen_img"
xml_dir = "/home/gytang/huangwen/data/zheyun/lowamt/fen_xml"
tgt_path = "/home/gytang/huangwen/data/zheyun/lowamt/fen_nolabel/xml/"
xml_lst = os.listdir(xml_dir)
for info in xml_lst[:100]:
    info = os.path.basename(info)
    with open(os.path.join(xml_dir, info), 'r') as f:
        # class_rst_dict = defaultdict(list)
        # # we fill the data path not label path in the class_rst_dict. It is used for separation
        # filled_value = filled_file_name if filled_file_name is not None else file_path
        # with open(file_path, 'r') as f:
        print(info)
        img = cv2.imread(os.path.join(img_dir, info.split('.')[0] + '.jpg'))
        try:
            height, width, _ = img.shape
        except:
            continue
        xml_tree = ET.parse(f)
        root = xml_tree.getroot()
        class_rst_lst = {}
        for_direction = 0
        obj_lst = root.findall("object")
        for obj_id, obj in enumerate(obj_lst):
            cls = obj.find('name').text
            #             if cls[0]!='￥' and '.' in cls:
            #                 print(cls)
            label = cls.replace(".", "")
            class_rst_lst['name'] = label
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            #             if cls[0]!='￥' and '.' in cls:
            #                 print(cls)
            crop_img = img[:, :xmin - 2, :]
            print(xmin)
            print(img.shape)
            print(crop_img.shape)
            cv2.imshow("", crop_img)
            cv2.waitKey(900)

            cv2.imwrite(os.path.join("/home/gytang/huangwen/data/zheyun/lowamt/fen_nolabel/image/", info), crop_img)
        bndbox = obj.find("bndbox")
        content = xml_template_content.replace(width_template, str(width)). \
            replace(height_template, str(height))
        with open(os.path.join(tgt_path, info), 'w') as f:
            f.write(content)
