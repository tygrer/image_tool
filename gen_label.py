#coding=gbk
import codecs

# import sys
# from xlrd import open_workbook # xlrd���ڶ�ȡxld
# # import xlwt  # ����д��xls
# workbook = open_workbook(r'/home/gytang/Downloads/ȫ����վ��׼վ���ֵ�.xls',encoding_override='utf-8')  # ��xls�ļ�
# sheet_name= workbook.sheet_names()  # ��ӡ����sheet���ƣ��Ǹ��б�
# sheet = workbook.sheet_by_index(0)  # ����sheet������ȡsheet�е���������
# sheet1= workbook.sheet_by_name('Sheet1')  # ����sheet���ƶ�ȡsheet�е���������
# print(sheet.name, sheet.nrows, sheet.ncols)  # sheet�����ơ�����������
# content = sheet.col_values(7)  # ����������
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

    # ���д�д��ĸ
    # for i in range(3):
    #     chr(random.randint(65,91))
    title_train = ['K','G','D','T','Z','C']
    label = ''.join(random.sample(title_train, 1))
    # ����Сд��ĸ
    for i in range(4):
        label = label+chr(random.randint(48,57))
    print(label)
    f.write(":"+label+'\n')
    #����
    label=''
    for i in range(14):
        label = label+chr(random.randint(48,57))
    label = label+chr(random.randint(65, 90))
    for i in range(6):
        label = label+chr(random.randint(48,57))
    print(label)
    f.write(":"+label+'\n')

    type_train = ['�¿յ�Ӳ��','�¿յ�����','һ����','������']
    label = ''.join(random.sample(type_train, 1))
    print(label)
    f.write(":"+label+'\n')

    nian = ['2016','2017','2018','2019']
    yue = ['01','02','03','04','05','06','07','08','09','10','11','12']
    ri = ['01','02','03','04','05','06','07','08','09']
    for i in range(10,31):
        ri.append(str(i))
    label = ''.join(random.sample(nian, 1))
    label = label+('��')
    label = label+(random.sample(yue, 1))[0]
    label = label+('��')
    label = label+(random.sample(ri, 1))[0]
    label = label+('��')
    print(label)
    f.write(":"+label+'\n')

    #money
    label = '��'+str(random.randint(1,1200))
    label = label+('.0Ԫ') if random.random()<0.5 else label+('.5Ԫ')
    print(label)
    f.write(":"+label+'\n')

f.close()
# ��������
# for i in range(48,58):
#     character.append(chr(i))
# print(character)