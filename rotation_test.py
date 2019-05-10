from PIL import Image
import cv2
# # original image
# img = Image.open('/home/gytang/project/dataset/traintickets/crop_image/火车票-fq-001_5.jpg')
# # converted to have an alpha layer
# im2 = img.convert('RGBA')
# # rotated image
# rot = im2.rotate(22.2, expand=1)
# width,height = rot.size
# #a white image same size as rotated image
# fff = Image.open('/home/gytang/Desktop/background_hcp/hc-06.jpg')
# fff_resize = fff.resize((width,height), Image.ANTIALIAS)
# #imgr = rotate_image(fff, 22)
# # create a composite image using the alpha layer of rot as a mask
# out = Image.composite(rot, fff_resize, rot)
# # save your work (converting back to mode='1' or whatever..)
# out.convert(img.mode).save('test2.png')
# out.show()

import numpy as np
img = cv2.imread('/home/gytang/project/dataset/traintickets/crop_image/火车票-fq-001_5.jpg')
h,w,c = img.shape
# cv2.namedWindow("",0)
# cv2.resizeWindow("",300,100)
cv2.imshow("", img)
bkimg = cv2.imread('/home/gytang/Desktop/background_hcp/hc-06.jpg')
cv2.waitKey(1000)
src = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
dst = np.float32([[6, 6], [w-6, 6], [3, h-1], [w-3, h-1]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, m, (0,0), bkimg, borderMode=cv2.BORDER_REPLICATE)
#result = cv2.warpPerspective(img, m, (0,0))
#result = cv2.resize(img,(img.shape[1], int(img.shape[0]*0.6)))
# cv2.namedWindow("1",0)
# cv2.resizeWindow("1",300,100)
cv2.imshow("1", result)

cv2.waitKey(100000)



