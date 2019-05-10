import random
import char_utils
dir_txt = '/home/gytang/project/traintickets/'
f = open(dir_txt+'traintickets.txt','r')
ff = f.readlines()
random.shuffle (ff)
f.close()
f_new = open(dir_txt+'traintickets_info.txt','w')
f_c = open('/home/gytang/project/rcnn/words/chinese_word_list.txt','a')
k = char_utils.UNKNOWN_INDEX
for line in ff[int(len(ff)*0.9):]:
     if line.split(':')[1] == '\n'or '*' in line.split(':')[1]:
          continue
     label=[]
     #if "¥" in line:
     # line = line.replace("¥", "￥")
     # line  = line.replace("，", "")
     #print(char_utils.UNKNOWN_INDEX)
     for w in line.split(':')[1][:-1]:

          if char_utils.encode_maps.get(w,k)==8225:
               f_c.write(w)
               print(line, w)
          #label.append(char_utils.encode_maps.get(w, char_utils.UNKNOWN_INDEX))

     # print("UNKNOWN_INDEX:", char_utils.UNKNOWN_INDEX)
     # print(label)
     # print(line.strip().split(':')[1])
     #f_new.write("/iqubicdata/workspace/sunwei/datasets/YangGuang/planetickets/crop_image/"+line)
     f_new.write(line)
f_c.close()
f_new.close()
