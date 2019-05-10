import glob,os
dir = '/home/gytang/project/dataset/2019defp/all/'
# txt_lst = glob.glob(os.path.join(dir,'*.txt'))
all = open(dir+'all.txt','w')
jpg_lst = glob.glob(dir+'*.jpg')
jie = int(len(jpg_lst)*0.9)
# train = open(dir+'train.txt','r')
# val =  open(dir+'val.txt','r')
train = open(dir+'train.txt','w')
val = open(dir+'val.txt','w')
train_lst = []
val_lst = []
for i in jpg_lst[:jie]:
    if os.path.exists(i) and os.path.exists(os.path.splitext(i)[0]+'.xml'):
        i = os.path.basename(os.path.splitext(i)[0])
        print(i)
        train.write(i+'\n')
        all.write(i+'\n')
for i in jpg_lst[jie:]:
    if os.path.exists(i) and os.path.exists(os.path.splitext(i)[0]+'.xml'):
        i = os.path.basename(os.path.splitext(i)[0])
        val.write(i+'\n')
        all.write(i+'\n')
all.close()
train.close()
val.close()
