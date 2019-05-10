img_dir = '/home/gytang/project/myrcnn/rcnn/background/'
from PIL import Image

import os,cv2
for i in os.listdir(img_dir):
    image = Image.open(os.path.join(img_dir, i))
    print(i,image.mode)