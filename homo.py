import cv2,random
import numpy as np

template = {'goumaifang': [208.5, 357.0],
            'xiaoshoufang': [206.0, 991.5],
            'mimaqu': [1193.0, 351.5],
            'beizhu': [1198.5, 988.5],
            'guigexinghao': [738.5, 471.0],
            'shuliang': [1044.0, 469.5],
            'jine': [1432.5, 467.5],
            'shoukuanren': [256.0, 1100.5],
            'fuhe': [736.0, 1102.0],
            'kaipiaoren': [1130.0, 1100.0],
            'xiaoxie': [1486.5, 860.0],
            'jiashuiheji': [401.5, 867.5]}
dst_pts = []
flage = 0

for key, pix in template.items():
    dst_pts.append(pix)
    dst_pts.append(pix)
# for key, pix in template.items():
#     dst_pts.append(pix)

image = cv2.imread("/home/gytang/vat2.jpg")

detection = {'goumaifang': [[676.5, 281.0], [436.0, 136.5]],
             'xiaoshoufang': [[147.5, 134.5], [671.0, 564.5]],
             'mimaqu': [[435.0, 592.0], [1125.0, 276.0]],
             'beizhu': [[144.5, 591.5], [1124.0, 566.0]],
             'guigexinghao': [[384.0, 380.5], [916.0, 328.0]],
             'shuliang': [[383.5, 522.0], [1057.0, 328.5]],
             'jine': [[383.5, 703.0], [1237.5, 328.5]],
             'shoukuanren': [[99.0, 158.5], [694.0, 613.0]],
             'fuhe': [[909.5, 614.5], [97.5, 378.5]],
             'kaipiaoren': [[1090.5, 615.0], [97.0, 556.0]],
             'xiaoxie': [],
             'jiashuiheji': [[205.0, 224.0], [759.0, 507.0]],}
src_pts = []
src_pts1 = []
flage = 0
for key, pix in detection.items():
    if len(pix) == 2:
        #if random.random()<0.5:
        src_pts.append(pix[0])
        #src_pts1.append(pix[1])
        #cv2.circle(image, (int(pix[0][0]), int(pix[0][1])), 5, (255, 255, 0), 5)
    #else:
        src_pts.append(pix[1])
        #src_pts1.append(pix[0])
        cv2.circle(image, (int(pix[0][0]), int(pix[0][1])), 5, (255, 255, 0), 5)
        cv2.circle(image, (int(pix[1][0]), int(pix[1][1])), 5, (255, 255, 0), 5)
    else:
        src_pts.append([0,0])
        src_pts.append([0,0])
        cv2.circle(image, (0, 0), 5, (255, 255, 0), 5)

cv2.imshow("",image)
cv2.waitKey(1000)
M, mask = cv2.findHomography(np.array(dst_pts), np.array(src_pts), cv2.RANSAC, ransacReprojThreshold=15,maxIters=100)
matchesMask = mask.ravel().tolist()
for p_id, point in enumerate(src_pts):
    if matchesMask[p_id] == 1:
        cv2.circle(image, (int(point[0]), int(point[1])), 10, (255,0,0), 8)
cv2.imshow("",image)
cv2.waitKey(100000)