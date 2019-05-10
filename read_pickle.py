import pickle,os
path ="/home/gytang/project/dataset/yg_bak_0722/pre_air_clean/pickles/"
save_pkl = "/home/gytang/project/dataset/yg_bak_0722/air_val.pkl"
img_lst = []
label = []
for i in os.listdir(path):
    f = open(os.path.join(path, i), 'rb')
    ppp = pickle.load(f)
    img_lst.append(os.path.splitext(i)[0]+'.jpg')

    if ppp.get('jp-hbh') is not None:
        ppp['jp-hbh'] = ppp.get('jp-hbh')[0]
    if ppp.get('jp-rq') is not None:
        ppp['jp-rq'] = ppp.get('jp-rq')[0]
    if ppp.get('jp-ddd') is not None:
        ppp['jp-ddd'] = ppp.get('jp-ddd')[0]

    label.append(ppp)
    print(ppp)
    # for pi in ppp:
label_tuple = (img_lst, label)
f1 = open(save_pkl,"wb")
pickle.dump(label_tuple, f1)
# f = open("/home/gytang/project/dataset/yg_bak_0722/invoice_val.pkl", 'rb')
# ppp = pickle.load(f)
# print(ppp)