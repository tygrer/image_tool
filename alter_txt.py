import random
import char_utils
dir_txt = './'
f = open(dir_txt+'defp_2.19_val_crop.txt','r')
ff = f.readlines()
random.shuffle (ff)
f.close()
f_new = open(dir_txt+'defp_2.19_val.txt','w')
for line in ff:
     if line.split(':')[1] == '\n'or '*' in line.split(':')[1]:
          continue
     label=[]
     for w in line.split(':')[1][:-1]:
          if char_utils.encode_maps.get(w, char_utils.UNKNOWN_INDEX)==4165:
               print(line,w)
          label.append(char_utils.encode_maps.get(w, char_utils.UNKNOWN_INDEX))

     # print("UNKNOWN_INDEX:", char_utils.UNKNOWN_INDEX)
     # print(label)
     # print(line.strip().split(':')[1])
     f_new.write(line)

f_new.close()



#
# import random,os
# import char_utils
# dir_txt = './'
# f = open(dir_txt+'defp_2.19_train_new.txt','r')
# ff = f.readlines()
# # random.shuffle (ff)
# f.close()
# # f_new = open(dir_txt+'defp_2.19_train.txt','w')
# for line in ff:
#      filename = os.path.basename(line.split(':')[0])
#      if not os.path.exists(os.path.join('/home/gytang/project/dataset/2019.1.30/train/',filename)):
#           print(filename)


     # print("UNKNOWN_INDEX:", char_utils.UNKNOWN_INDEX)
     # print(label)
     # print(line.strip().split(':')[1])
#      f_new.write(line)
#
# f_new.close()