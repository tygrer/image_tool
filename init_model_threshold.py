# -*- coding: UTF-8 -*-
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
def init_predict_mode(demo_dir):
    X_train = np.load("./x169.npy")
    Y_train = np.load("./y169.npy")
    img_list = os.listdir(demo_dir)
    for i in range(169,len(img_list),1):
        image = cv2.imread(demo_dir + img_list[i])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])

        #plt.hist(gray.flatten(),bins=256, normed=1, edgecolor='None', facecolor='red')
        #plt.show()
        cv2.imshow('', gray)
        cv2.waitKey(10)

        mid = 220
        while(1):
            (_, thresh) = cv2.threshold(gray, mid, gray.max(), cv2.THRESH_BINARY)
            cv2.imshow('', thresh)
            cv2.waitKey(10)
            input_mid = input('请输入阈值:')

            if input_mid is "":

                if X_train is None:
                    X_train = hist
                    Y_train = mid
                else:
                    X_train = np.hstack((X_train, hist))
                    Y_train = np.hstack((Y_train, mid))
                break

            else:
                mid = int(input_mid)
            np.save("x"+str(i)+".npy", X_train)
            np.save("y"+str(i)+".npy", Y_train)
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    max_depth = 20
    model_g = GradientBoostingRegressor(n_estimators=50, loss='huber', max_depth=max_depth)
    model_g.fit(X_train, Y_train)
    from sklearn.externals import joblib
    joblib.dump(model_g, "thresh.pkl")
    return model_g

def train_predict_mode():
    X_train = np.load("./x561.npy")
    Y_train = np.load("./y561.npy")

    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    max_depth = 20
    model_g = GradientBoostingRegressor(n_estimators=50, loss='huber', max_depth=max_depth)
    model_g.fit(np.transpose(X_train), Y_train)
    from sklearn.externals import joblib
    joblib.dump(model_g, "thresh.pkl")
    return model_g


if __name__ == '__main__':
    demo_dir = '/home/tanggy/Tencent_corrected_for_ctc_0527_181_img_slice/'
    #demo_dir = '/home/tanggy/guiyu/TEST/'
    #init_predict_mode(demo_dir)
    train_predict_mode()