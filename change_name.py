import os
import shutil

in_path = "/media/gytang/My Passport3/RESIDE/train/"
data_dir = os.listdir(in_path)
data_dir.sort()
out_path = "/media/gytang/My Passport3/RESIDE/val"
for i, cls_i in enumerate(data_dir[:10]):
    # for j in os.listdir(os.path.join(in_path, cls_i))[:30]:
    print(cls_i,"---->",str(i))
    shutil.copy(os.path.join(in_path,cls_i), os.path.join(out_path,str(i))+'.h5')
