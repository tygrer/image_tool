import cv2,random
import numpy as np
def PepperandSalt(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    NoiseImg.flags.writeable = True
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.uniform(0, 1) <= 0.5:
            NoiseImg[randX, randY] = 219
        else:
            NoiseImg[randX, randY] = 100
    return NoiseImg

# def PointSalt(src, point, percetage):
#     NoiseImg = src
#     # NoiseNum = int(percetage * src.shape[0] * src.shape[1])
#     edge_ratio = 1
#     NoiseImg.flags.writeable = True
#     ymin = min(max(point[1] - edge_ratio, 0), src.shape[1])
#     ymax = min(max(point[1] + edge_ratio, ymin), src.shape[1])
#     xmin = min(max(point[0] - edge_ratio, 0), src.shape[0])
#     xmax = min(max(point[0] + edge_ratio, xmin), src.shape[0])
#     #NoiseNum = int((ymax - ymin) * (xmax - xmin) * percetage)
#     NoiseNum=2
#     for i in range(NoiseNum):
#         randX = random.randint(xmin, xmax-1)
#         randY = random.randint(ymin, ymax-1)
#         #if random.uniform(0, 1) <= 0.5:
#         if NoiseImg[point[0], point[1], 0] == NoiseImg[point[0], point[1], 1] and NoiseImg[point[0], point[1], 0] == NoiseImg[point[0], point[1], 2]:
#             value = random.randint(NoiseImg[point[0], point[1], 0] - 2,
#                                                        NoiseImg[point[0], point[1], 0] + 2)
#             NoiseImg[randX, randY, 0] = value
#             NoiseImg[randX, randY, 1] = value
#             NoiseImg[randX, randY, 2] = value
#         else:
#             if random.uniform(0, 1) <= 0.33:
#                 NoiseImg[randX, randY, 0] = random.randint(NoiseImg[point[0],point[1], 0] - 2, NoiseImg[point[0],point[1], 0] + 2)
#                 NoiseImg[randX, randY, 1] = NoiseImg[point[0],point[1], 1]
#                 NoiseImg[randX, randY, 2] = NoiseImg[point[0], point[1], 2]
#             elif random.uniform(0, 1) > 0.33 and random.uniform(0, 1) <= 0.66:
#                 NoiseImg[randX, randY, 0] = NoiseImg[point[0],point[1], 0]
#                 NoiseImg[randX, randY, 1] = random.randint(NoiseImg[point[0],point[1], 1] - 2, NoiseImg[point[0],point[1], 1] + 2)
#                 NoiseImg[randX, randY, 2] = NoiseImg[point[0], point[1], 2]
#             else:
#                 NoiseImg[randX, randY, 0] = NoiseImg[point[0], point[1], 0]
#                 NoiseImg[randX, randY, 1] = NoiseImg[point[0], point[1], 1]
#                 NoiseImg[randX, randY, 2] = random.randint(NoiseImg[point[0], point[1], 2] - 2,
#                                                            NoiseImg[point[0], point[1], 2] + 2)
#     return NoiseImg

# import os
# for i in os.listdir('./image/'):
#     img = cv2.imread(os.path.join("./image/", i))
#     cv2.imshow("ee",img)
#     cv2.waitKey(5000)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
#     y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
#
#     absX = cv2.convertScaleAbs(x)  # 转回uint8
#     absY = cv2.convertScaleAbs(y)
#
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     list_dst = np.where(dst > dst.mean()*2)
#     for i_id, zuobiao in enumerate(list_dst[0]):
#         x = list_dst[0][i_id]
#         y = list_dst[1][i_id]
#         img = PointSalt(img,(x,y),0.01)
#     cv2.imshow("Result", img)
#     cv2.waitKey(2000)
def PointSalt(src):
    NoiseImg = src
    # NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    edge_ratio = 2
    NoiseNum=100
    NoiseImg.flags.writeable = True
    gray = cv2.cvtColor(NoiseImg, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    list_dst = np.where(dst > dst.mean() * 2)

    list_point = [(list_dst[0][i],list_dst[1][i]) for i in range(len(list_dst))]
    random.shuffle(list_point)
    for i_id, zuobiao in enumerate(list_point[:int(len(list_point)/2)]):
        x = list_point[i_id][0]
        y = list_point[i_id][1]
        point = (x,y)
        ymin = min(max(point[1] - edge_ratio, 0), src.shape[1])
        ymax = min(max(point[1] + edge_ratio, ymin), src.shape[1])
        xmin = min(max(point[0] - edge_ratio, 0), src.shape[0])
        xmax = min(max(point[0] + edge_ratio, xmin), src.shape[0])
        #NoiseNum = int((ymax - ymin) * (xmax - xmin) * percetage)
        for i in range(NoiseNum):
            randX = random.randint(xmin, xmax-1)
            randY = random.randint(ymin, ymax-1)
            #if random.uniform(0, 1) <= 0.5:
            if NoiseImg[point[0], point[1], 0] == NoiseImg[point[0], point[1], 1] and NoiseImg[point[0], point[1], 0] == NoiseImg[point[0], point[1], 2]:
                value = random.randint(NoiseImg[point[0], point[1], 0] - 20,
                                                           NoiseImg[point[0], point[1], 0] + 20)
                value = min(255, max(value, 0))
                NoiseImg[randX, randY, 0] = value
                NoiseImg[randX, randY, 1] = value
                NoiseImg[randX, randY, 2] = value
            else:
                if random.uniform(0, 1) <= 0.33:
                    NoiseImg[randX, randY, 0] = random.randint(NoiseImg[point[0],point[1], 0] - 2, NoiseImg[point[0],point[1], 0] + 2)
                    NoiseImg[randX, randY, 1] = NoiseImg[point[0],point[1], 1]
                    NoiseImg[randX, randY, 2] = NoiseImg[point[0], point[1], 2]
                elif random.uniform(0, 1) > 0.33 and random.uniform(0, 1) <= 0.66:
                    NoiseImg[randX, randY, 0] = NoiseImg[point[0],point[1], 0]
                    NoiseImg[randX, randY, 1] = random.randint(NoiseImg[point[0],point[1], 1] - 2, NoiseImg[point[0],point[1], 1] + 2)
                    NoiseImg[randX, randY, 2] = NoiseImg[point[0], point[1], 2]
                else:
                    NoiseImg[randX, randY, 0] = NoiseImg[point[0], point[1], 0]
                    NoiseImg[randX, randY, 1] = NoiseImg[point[0], point[1], 1]
                    NoiseImg[randX, randY, 2] = random.randint(NoiseImg[point[0], point[1], 2] - 2,
                                                               NoiseImg[point[0], point[1], 2] + 2)
    return NoiseImg
import os
for i in os.listdir('./image/'):
    img = cv2.imread(os.path.join("./image/", i))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #img = np.array([img,img,img])
    b = np.reshape(img, (img.shape[0], img.shape[1], 1))
    img = np.concatenate([b, b,  b], 2)
    img = cv2.resize(img,(1024,1024))
    cv2.imshow("ee",img)
    cv2.waitKey(500)
    img = PointSalt(img)
    cv2.imshow("Result", img)
    cv2.waitKey(2000)



# cv2.destroyAllWindows()
# image = PepperandSalt(img, 0.05)
# image = cv2.GaussianBlur(image, (3,3), 0)
# cv2.imshow("",image)
# cv2.waitKey(0)
# cv2.imshow("absX", absX)
#     # cv2.imshow("absY", absY)
#     #img = cv2.GaussianBlur(img, (3,3), 0) 9
#     # d = cv2.getTrackbarPos("d","image")
#     # sigmaColor = cv2.getTrackbarPos("sigmaColor","image")
#     # sigmaSpace = cv2.getTrackbarPos("sigmaSpace","image")
#     # img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
#     #img = cv2.medianBlur(img, 3)
#     # img = cv2.bilateralFilter(img,9,75,75)

    # kernel = np.array([[0, -1, 0], [0, 3, 0], [0, -1, 0]], np.float32) #锐化
    # img = cv2.filter2D(img, -1, kernel=kernel)
    # cv2.imshow("Result", dst)
    #
    # cv2.waitKey(1000)