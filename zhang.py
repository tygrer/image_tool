import cv2

zhang = cv2.imread('/home/gytang/dataset/zhang/2.png')
zh, zw, _ = zhang.shape
#zhang = cv2.GaussianBlur(zhang, ksize=(5, 5), sigmaX=0, sigmaY=0)
#cv2.imshow("nnff", zhang)
#cv2.waitKey(1000)
#cv2.imshow("", zhang)
#cv2.waitKey(1000)
(_, thresh) = cv2.threshold(zhang[:, :, 0], 215, zhang[:, :, 0].max(), cv2.THRESH_BINARY)
idx_lst = []
cv2.imshow("1", thresh)
cv2.waitKey(1000)
for i in range(thresh.shape[0]):
    for j in range(thresh.shape[1]):
        if thresh[i][j] == 0:
            idx_lst.append([i, j])
zhang = cv2.imread(FLAGS.zhang_dir)
res = cv2.addWeighted(img1, i, zhang, (1-i), 0)

