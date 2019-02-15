import os
import shutil

in_path = "/workspace/data/ILSVRC2015/Data/CLS-LOC/train"
data_dir = os.listdir(in_path)
data_dir.sort()
out_path = "/workspace/tanggy/small_image_set/"
for i, cls_i in enumerate(data_dir):
    if cls_i == 'labels.txt':
        continue
    os.makedirs(os.path.join(out_path,str(i+1)))
    for j in os.listdir(os.path.join(in_path, cls_i))[:30]:
        shutil.copy(os.path.join(in_path,cls_i,j), os.path.join(out_path,str(i+1),j))

