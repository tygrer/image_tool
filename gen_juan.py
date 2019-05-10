import random
import char_utils
dir_txt = './'
import pickle
famlyf = pickle.load(open(dir_txt + 'companyBizInfo.from58.pkl','rb'))
with open(dir_txt+'juan_info.txt','w') as f:
    for i in range(10000):
        len_nums = 12
        if random.random()<0.5:
            line = ":" + "发票代码"###
        else:
            line = ":"
        for j in range(len_nums):
            line = line+str(random.randint(0,12))
        line = line+'\n'
        print(line)
        f.write(line)

        len_nums = 8
        if random.random()<0.5:
            line = ":" + "发票号码"###
        else:
            line = ":"
        for j in range(len_nums):
            line = line+str(random.randint(0,9))
        line = line+'\n'
        print(line)
        f.write(line)

        if random.random()<0.5:
            line = ":" + "机打号码"###
        else:
            line = ":"
        for j in range(len_nums):
            line = line+str(random.randint(0,9))
        line = line+'\n'
        print(line)
        f.write(line)

        if random.random()<0.5:
            line = ":" + "机器号码"###
        else:
            line = ":"
        for j in range(len_nums):
            line = line+str(random.randint(0,9))
        line = line+'\n'
        print(line)
        f.write(line)

        if random.random()<0.5:
            label = ":" + "销售方名称"  ###
        else:
            label = ":"
        # char_lst = list(char_utils.encode_maps.keys())
        # label_len = random.randint(12, 27)
        # # label = ''.join(random.sample(char_lst, label_len))\
        # for i in range(label_len):
        #     while (1):
        #         zi = random.randint(0, len(char_lst) - 1)
        #         if u'\u4e00' <= char_lst[zi] <= u'\u9fff':
        #             break
        #         else:
        #             continue
        #     label = label + char_lst[zi]
        famly_no = []
        while (len(famly_no) < 8):
            famly_lst = list(famlyf.keys())
            famly_no = famly_lst[random.randint(0, len(famly_lst))]

        if "【" in famly_no:
            label += famly_no.split('】')[1]
        else:
            label += famly_no
        f.write(label+'\n')
        print(label+'\n')

        if random.random() < 0.5:
            line = ":" + "纳税人识别号"  ###
        else:
            line = ":"
        len_nums = random.randint(5, 17)
        for j in range(len_nums):
            line = line + str(random.randint(0,9))
        for j in range(18-len_nums):
            line = line + str(random.randint(0,9))
        print(line+'\n')
        f.write(line+"\n")

        if random.random() < 0.5:
            label = ":" + "开票日期"
        else:
            label = ":"
        if random.random()<0.4:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = label + ''.join(random.sample(nian, 1))
            label = label + ('-')
            label = label + (random.sample(yue, 1))[0]
            label = label + ('-')
            label = label + (random.sample(ri, 1))[0]
        else:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = label + ''.join(random.sample(nian, 1))
            label = label + (random.sample(yue, 1))[0]
            label = label + (random.sample(ri, 1))[0]
        print(label+'\n')
        f.write(label + '\n')

        if random.random() < 0.5:
            label = ":" + "开票日期"
        else:
            label = ":"
        if random.random()<0.4:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = label + ''.join(random.sample(nian, 1))
            label = label + ('-')
            label = label + (random.sample(yue, 1))[0]
            label = label + ('-')
            label = label + (random.sample(ri, 1))[0]
        else:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = label + ''.join(random.sample(nian, 1))
            label = label + (random.sample(yue, 1))[0]
            label = label + (random.sample(ri, 1))[0]
        print(label+'\n')
        f.write(label + '\n')

        if random.random() < 0.5:
            label = ":" + "开票日期"
        else:
            label = ":"
        if random.random()<0.4:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = label + ''.join(random.sample(nian, 1))
            label = label + ('-')
            label = label + (random.sample(yue, 1))[0]
            label = label + ('-')
            label = label + (random.sample(ri, 1))[0]
        else:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = label + ''.join(random.sample(nian, 1))
            label = label + (random.sample(yue, 1))[0]
            label = label + (random.sample(ri, 1))[0]
        print(label+'\n')
        f.write(label + '\n')
        if random.random() < 0.5:
            line = ":" + "纳税人识别号"  ###
        else:
            line = ":"
        len_nums = random.randint(5, 18)
        for j in range(len_nums):
            line = line + str(random.randint(0,9))
        for j in range(18-len_nums):
            if random.random() < 0.5:
                line = line + chr(random.randint(65,  91))
            else:
                line = line + str(random.randint(0, 9))
        print(line+'\n')

        f.write(line + '\n')

        if random.random() < 0.5:
            line = ":" + "购买方名称"
        else:
            line = ":"
        # char_lst = list(char_utils.encode_maps.keys())
        # label_len = random.randint(10, 27)
        # label = ''.join(random.sample(char_lst, label_len))\

        famly_no = []
        while (len(famly_no) < 8):
            famly_lst = list(famlyf.keys())
            famly_no = famly_lst[random.randint(0, len(famly_lst))]

        if "【" in famly_no:
            line += famly_no.split('】')[1]
        else:
            line += famly_no
        print(line+'\n')
        f.write(line+ '\n')

        if random.random() < 0.5:
            line = ":" + "纳税人识别号"  ###
        else:
            line = ":"
        len_nums = random.randint(5, 18)
        for j in range(len_nums):
            line = line + str(random.randint(0, 9))
        for j in range(18 - len_nums):
            if random.random() < 0.5:
                line = line + chr(random.randint(65,  91))
            else:
                line = line + str(random.randint(0, 9))
        print(line+'\n')
        f.write(line + '\n')

        char_lst = list(char_utils.encode_maps.keys())
        label_len = random.randint(10, 27)
        # label = ''.join(random.sample(char_lst, label_len))\
        label = ""
        for i in range(label_len):
            while (1):
                zi = random.randint(0, len(char_lst) - 1)
                if u'\u4e00' <= char_lst[zi] <= u'\u9fff':
                    break
                else:
                    continue
            label = label + char_lst[zi]
        print(label+'\n')
        f.write(":" + label + '\n')

        # namef = open(dir_txt + 'full_names_clean.csv', 'r')
        # name_lst = namef.readlines()
        # name_no = random.randint(0, len(name_lst))
        # print(":"+ name_lst[name_no])
        # f.write(":" + name_lst[name_no])

        # char_lst = list(char_utils.encode_maps.keys())
        # for j in range(len_nums):
        #     line = line + str(random.randint(5, 17))
        label_len = random.randint(8,9)
        # label = ''.join(random.sample(char_lst, label_len))\
        label = ""
        for i in range(label_len):
            while(1):
                zi = random.randint(0, len(char_lst) - 1)
                if u'\u4e00' <= char_lst[zi] <= u'\u9fff':
                    break
                else:
                    continue
            label = label + char_lst[zi]
        print(":" + label+'\n')
        f.write(":" + label + '\n')

        label_len = random.randint(1,9)

        # label = ''.join(random.sample(char_lst, label_len))\
        label = ""
        for i in range(label_len):
            while(1):
                zi = random.randint(0, len(char_lst) - 1)
                if u'\u4e00' <= char_lst[zi] <= u'\u9fff':
                    break
                else:
                    continue
            label = label + char_lst[zi]
        print(label+'\n')
        f.write(":" + label + '\n')


        # money

        label = str(random.randint(5, 10000)) + '.' + str(random.randint(0, 9)) + str(random.randint(0, 9))

        print(label)
        f.write(":" + label + '\n')

        # money
        if random.random() < 0.5:
            label = "合计金额(小写)"
        else:
            label = ""
        if random.random()<0.5:
            label = label + '￥'+str(random.randint(5, 10000)) + '.' + str(random.randint(0, 9)) + str(random.randint(0, 9))
        else:
            label  = label + str(random.randint(5, 10000)) + '.' + str(random.randint(0, 9)) + str(random.randint(0, 9))
        print(label)
        f.write(":" + label + '\n')

        #jiaoyan
        if random.random() < 0.5:
            label = "校验码"
        else:
            label = ""
        len_nums = 20
        for j in range(len_nums):
            label = label + str(random.randint(0, 9))
        print(label)
        f.write(":" + label + "\n")
