import os
for i in os.listdir("/home/gytang/project/dataset/yg_bak_0722/mix_train"):
    name = str(int(i.split('.')[0][3:])+500)

    os.rename("/home/gytang/project/dataset/yg_bak_0722/mix_train/"+i, "/home/gytang/project/dataset/yg_bak_0722/mix_train/"+'mix'+name+'.jpg')