import random
import char_utils
#dir_txt = '/home/gytang/project/dataset/yg_railway_airplane/air/'
dir_txt = './'
fr = open(dir_txt+'plane_info.txt','r')
fr_lst = fr.readlines()
fr.close()
fw = open('./plane_info.txt','w')
random.shuffle(fr_lst)
#f_c = open('/home/gytang/project/rcnn/words/chinese_word_list.txt','a')
k = char_utils.UNKNOWN_INDEX
print(k)
for line in fr_lst:
    flag = 0
    if line.split(':')[-1] == '\n':
        continue
    for w in line.split(':')[-1][:-1]:
        if char_utils.encode_maps.get(w, char_utils.UNKNOWN_INDEX) == 4224:
            print(w)
            flag = 1
    if flag == 0:
        fw.write(line)

        #fw.write('/iqubicdata/workspace/sunwei/datasets/YangGuang/planetickets/crop_image2/' + line)
fw.close()
