import random
dir_txt = './'
with open(dir_txt+'defp.txt','w') as f:
    for i in range(100000):
        len_nums = random.randint(10,15)
        line = ":"+"发票代码"###
        for j in range(len_nums):
            line = line+str(random.randint(0,9))
        line = line+'\n'
        f.write(line)

        len_nums = random.randint(6,8)
        line = ":"+"发票号码"###
        for j in range(len_nums):
            line = line+str(random.randint(0,9))
        line = line+'\n'
        f.write(line)
        prs = random.random()
        if prs < 0.1:
            if random.random()<0.5:
                zwname = "壹佰元"
            else:
                zwname = "壹佰元整"
        elif prs >0.1 and prs<0.2:
            if random.random()<0.5:
                zwname = "贰佰元"
            else:
                zwname = "贰佰元整"
        elif prs >0.2 and prs<0.3:
            if random.random()<0.5:
                zwname = "伍拾元"
            else:
                zwname = "伍拾元整"
        elif prs >0.5 and prs<0.6:
            if random.random()<0.5:
                zwname = "壹拾元"
            else:
                zwname = "壹拾元整"
        elif prs >0.3 and prs<0.4:
            if random.random() < 0.7:
                zwname = "拾元整"
            else:
                zwname = "拾元"
        elif prs >0.4 and prs<0.5:
            zwname = "壹元整"
        elif prs >0.6 and prs<0.7:
            if random.random()<0.5:
                zwname = "贰拾元"
            else:
                zwname = "贰拾元整"
        elif prs >0.7 and prs<0.8:
            zwname = "贰元整"
        elif prs >0.8 and prs<0.9:
            zwname = "伍元整"
        else:
            if random.random()<0.5:
                zwname = "壹佰元"
            else:
                zwname = "壹佰元整"
        line = ":"+zwname###
        line = line+'\n'
        f.write(line)