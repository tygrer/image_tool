#coding=gbk
import codecs

# import sys
# from xlrd import open_workbook # xlrd用于读取xld
# # import xlwt  # 用于写入xls
# workbook = open_workbook(r'/home/gytang/Downloads/全国火车站标准站名字典.xls',encoding_override='utf-8')  # 打开xls文件
# sheet_name= workbook.sheet_names()  # 打印所有sheet名称，是个列表
# sheet = workbook.sheet_by_index(0)  # 根据sheet索引读取sheet中的所有内容
# sheet1= workbook.sheet_by_name('Sheet1')  # 根据sheet名称读取sheet中的所有内容
# print(sheet.name, sheet.nrows, sheet.ncols)  # sheet的名称、行数、列数
# content = sheet.col_values(7)  # 第六列内容
# print(content)
import pandas as pd
import char_utils
import random
f = open('/home/gytang/project/traintickets/traintickets.txt','w')
IO = '/home/gytang/Downloads/train_station.csv'
fy = codecs.open(IO, 'r', 'utf-8')
s = fy.readlines()
random.shuffle(s)
fy.close()
for j in range(10000):
    if j%len(s) == 0:
        random.shuffle(s)
    init_place = j%(len(s)-1)
    destination = j%(len(s)-1)+1
    f.write(":"+s[init_place].split(',')[1]+'\n')
    f.write(":"+s[destination].split(',')[1]+'\n')
    print(s[init_place].split(',')[1])
    print(s[destination].split(',')[1])

    char_lst = list(char_utils.encode_maps.keys())
    #name
    # if random.random()<0.4:
    label_len = random.randint(2, 3)
    #label = ''.join(random.sample(char_lst, label_len))\
    label = ""
    for i in range(label_len):
        while(1):
            zi = random.randint(0,len(char_lst)-1)
            if u'\u4e00' <= char_lst[zi] <= u'\u9fff':
                break
            else:
                continue
        label = label +char_lst[zi]

    print(label)
    f.write(":"+label+'\n')
    # else:
    #     label_len = random.randint(1, 3)
    #     label = ''.join(random.sample(char_lst, label_len))
    #No.

    # 所有大写字母
    # for i in range(3):
    #     chr(random.randint(65,91))
    title_train = ['K','G','D','T','Z','C']
    label = ''.join(random.sample(title_train, 1))
    # 所有小写字母
    for i in range(4):
        label = label+chr(random.randint(48,57))
    print(label)
    f.write(":"+label+'\n')
    #数字
    label=''
    for i in range(14):
        label = label+chr(random.randint(48,57))
    label = label+chr(random.randint(65, 90))
    for i in range(6):
        label = label+chr(random.randint(48,57))
    print(label)
    f.write(":"+label+'\n')

    type_train = ['新空调硬卧','新空调软卧','一等座','二等座']
    label = ''.join(random.sample(type_train, 1))
    print(label)
    f.write(":"+label+'\n')

    nian = ['2016','2017','2018','2019']
    yue = ['01','02','03','04','05','06','07','08','09','10','11','12']
    ri = ['01','02','03','04','05','06','07','08','09']
    for i in range(10,31):
        ri.append(str(i))
    label = ''.join(random.sample(nian, 1))
    label = label+('年')
    label = label+(random.sample(yue, 1))[0]
    label = label+('月')
    label = label+(random.sample(ri, 1))[0]
    label = label+('日')
    print(label)
    f.write(":"+label+'\n')

    #money
    label = '￥'+str(random.randint(1,1200))
    label = label+('.0元') if random.random()<0.5 else label+('.5元')
    print(label)
    f.write(":"+label+'\n')

f.close()
# 所有数字
# for i in range(48,58):
#     character.append(chr(i))
# print(character)