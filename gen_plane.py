import random
import char_utils
dir_txt = './'
fr = open(dir_txt+'airplane.txt','r')
fr_lst = fr.readlines()

fno = open(dir_txt+'planeno.txt','r')
fno_lst = fno.readlines()
char_lst = list(char_utils.encode_maps.keys())
with open(dir_txt+'plane_info.txt','w') as f:
    for i in range(30000):
        # name
        # if random.random()<0.4:
        # label_len = random.randint(2, 3)
        # # label = ''.join(random.sample(char_lst, label_len))\
        # label = ""
        # for i in range(label_len):
        #     while (1):
        #         zi = random.randint(0, len(char_lst) - 1)
        #         if u'\u4e00' <= char_lst[zi] <= u'\u9fff':
        #             break
        #         else:
        #             continue
        #     label = label + char_lst[zi]
        namef = open(dir_txt + 'full_names_clean.csv', 'r')
        name_lst = namef.readlines()
        name_no = random.randint(0, len(name_lst))
        print(":"+ name_lst[name_no])
        f.write(":" + name_lst[name_no])
        # print(label)
        # f.write(":" + label + '\n')

        #time
        if random.random()<0.8:
            nian = ['2016', '2017', '2018', '2019']
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = ''.join(random.sample(nian, 1))
            label = label + ('-')
            label = label + (random.sample(yue, 1))[0]
            label = label + ('-')
            label = label + (random.sample(ri, 1))[0]
        else:
            yue = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            ri = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
            for i in range(10, 31):
                ri.append(str(i))
            label = (random.sample(yue, 1))[0]
            label = label + ('-')
            label = label + (random.sample(ri, 1))[0]
        print(label)
        f.write(":" + label + '\n')

        #plane No.
        line = fno_lst[random.randint(0,len(fno_lst)-1)][:-1]
        label = line.split(' ')[0]
        no_lst = line.split('-')[1].split('、')
        label += no_lst[random.randint(0,len(no_lst)-1)]
        len_nums = 3
        for j in range(len_nums):
            label = label+str(random.randint(0,9))
        print(label)
        f.write(":"+label+ '\n')

        label=''
        #No.
        len_nums = [8,10,9,11,12,13,14,15]
        for j in range(random.sample(len_nums, 1)[0]):
            label = label+str(random.randint(0,9))
        print(label)
        f.write(":"+label+ '\n')
        T = ['T1','T2','T3']

        #addr
        rf = random.randint(0,len(fr_lst)-1)
        fromaddrstr = fr_lst[rf][:-1].split(' ')
        cond = random.random()
        if cond<=0.5:
            if fromaddrstr[-1][:2] == '黑龙' or fromaddrstr[-1][:2] == '内蒙':
                fromaddr = fromaddrstr[-1][3:]
            else:
                fromaddr = fromaddrstr[-1][2:]

        elif cond>0.5 and cond<=0.65:
            if fromaddrstr[-1][:2] == '黑龙' or fromaddrstr[-1][:2] == '内蒙':
                if len(fromaddrstr)==3:
                    fromaddr = fromaddrstr[-1][3:]+'/'+fromaddrstr[1]
                else:
                    fromaddr = fromaddrstr[-1][3:]
            else:
                if len(fromaddrstr) == 3:
                    fromaddr = fromaddrstr[-1][2:]+'/'+fromaddrstr[1]
                else:
                    fromaddr = fromaddrstr[-1][2:]

        elif cond>0.65 and cond<=0.8:
            if fromaddrstr[-1][:2] == '黑龙' or fromaddrstr[-1][:2] == '内蒙':
                if len(fromaddrstr) == 3:
                    fromaddr = fromaddrstr[-1][3:]+fromaddrstr[1]
                else:
                    fromaddr = fromaddrstr[-1][3:]
            else:
                if len(fromaddrstr) == 3:
                    fromaddr = fromaddrstr[-1][2:]+fromaddrstr[1]
                else:
                    fromaddr = fromaddrstr[-1][2:]

        elif cond>0.8 and cond<=1:

            if fromaddrstr[-1][:2] == '黑龙' or fromaddrstr[-1][:2] == '内蒙':
                fromaddr = (random.sample(T, 1))[0] + fromaddrstr[-1][3:]
            else:
                fromaddr = (random.sample(T, 1))[0] + fromaddrstr[-1][2:]
        else:
            print("exception:", fromaddrstr)
        print(fromaddr)
        f.write(":"+fromaddr+ '\n')
        rt = random.randint(0,len(fr_lst)-1)
        while(rt == rf):
            rt = random.randint(0,len(fr_lst)-1)
        toaddrstr = fr_lst[rt][:-1].split(' ')
        cond = random.random()
        if cond<=0.5:
            if toaddrstr[-1][:2] == '黑龙' or toaddrstr[-1][:2] == '内蒙':
                toaddr = toaddrstr[-1][3:]
            else:
                toaddr = toaddrstr[-1][2:]

        elif cond>0.5 and cond<=0.65:
            if toaddrstr[-1][:2] == '黑龙' or toaddrstr[-1][:2] == '内蒙':
                if fromaddrstr[-1][:2] == '黑龙' or fromaddrstr[-1][:2] == '内蒙':
                    if len(fromaddrstr) == 3:
                        toaddr = toaddrstr[-1][3:] + '/'+toaddrstr[1]
                    else:
                        toaddr = toaddrstr[-1][3:]
                else:
                    if len(fromaddrstr) == 3:
                        toaddr = toaddrstr[-1][2:] + '/'+toaddrstr[1]
                    else:
                        toaddr = toaddrstr[-1][2:]

        elif cond>0.65 and cond<=0.8:
            if fromaddrstr[-1][:2] == '黑龙' or fromaddrstr[-1][:2] == '内蒙':
                if len(fromaddrstr) == 3:
                    toaddr = toaddrstr[-1][3:] + toaddrstr[1]
                else:
                    toaddr = toaddrstr[-1][3:]
            else:
                if len(fromaddrstr) == 3:
                    toaddr = toaddrstr[-1][2:] + toaddrstr[1]
                else:
                    toaddr = toaddrstr[-1][2:]

        elif cond>0.8 and cond<=1:

            if toaddrstr[-1][:2] == '黑龙' or toaddrstr[-1][:2] == '内蒙':
                toaddr = (random.sample(T, 1))[0] + toaddrstr[-1][3:]
            else:
                toaddr = (random.sample(T, 1))[0] + toaddrstr[-1][2:]
        else:
            print("exception:", toaddrstr)

        print(toaddrstr,toaddr)
        f.write(":"+toaddr+ '\n')
        #money
        if random.random()<0.7:
            label = str(random.randint(500, 3500))+'.00'
        else:
            label = 'CNY'+str(random.randint(500, 3500)) + '.00'
        print(label)
        f.write(":"+label+'\n')

