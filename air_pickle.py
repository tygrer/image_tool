import os, pickle
for i in os.listdir("/home/gytang/Downloads/pickles/"):
     f = open(os.path.join('/home/gytang/Downloads/pickles/', 'fjp-045_1.pkl'), "rb")
     label = pickle.load(f)
     print(label)