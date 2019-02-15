import pickle
import zlib
import numpy as np
def pickle_compress_test():
    maritx = np.random.rand(5,5)
    maritx_byte = pickle.dumps(maritx)
    print(maritx)
    #print(zlib.compress(maritx_byte))
    fi=zlib.compress(maritx_byte)
    print(pickle.loads(zlib.decompress(fi)))
    with open("./save.darmx","wb") as f:
      a = pickle.dump(fi,f)

    with open("./save.darmx","rb") as f:
     full_data_lst_cmpr = pickle.load(f)
     a = pickle.loads(zlib.decompress(full_data_lst_cmpr))
    print(a)