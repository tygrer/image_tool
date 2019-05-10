import os,cv2,random

def rebuild_img(input_img, background_img):
    import cv2

    ticket_h, ticket_w = input_img.shape[:2]
    from pprint import pprint
    pprint(str(ticket_h) + ',' + str(ticket_w))
    # background = cv2.resize(background, (int(ticket_w * 3), int(ticket_h * 1.2)))
    # t1 = int(1*ticket_w)
    # t2 = int(2*ticket_w)

    background = cv2.resize(background_img, (int(ticket_w*1.2),int(ticket_h*1.3)))
    t1 = int(0.1*ticket_w)
    t2 = int(1.1*ticket_w)

    try:
        background[int(0.15 * ticket_h):int(1.15 * ticket_h), t1:t2, :] = input_img
    except:
        background[int(0.15 * ticket_h):int(1.15 * ticket_h)+1, t1:t2, :] = input_img
    #background[50:50+ticket_h, 50:51.50+ticket_w, :] = input_img
    return background

for i in os.listdir("/home/gytang/project/dataset/yg_bak_0722/roll_rotation_padding/"):
    input_img = cv2.imread(os.path.join("/home/gytang/project/dataset/yg_bak_0722/roll_rotation_padding",i))
    #background_img = "/home/gytang/Desktop/background_defp/Selection_004.jpg"
    height, width = input_img.shape[:2]
    # angle = random.randint(-5,5)
    # background = cv2.imread(background_img)
    # M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    # input_img = cv2.warpAffine(input_img, M, (int(width*1.05), int(height*1.05)), borderValue=(255, 255, 255))#)background, borderMode=cv2.BORDER_REPLICATE
    # #result = rebuild_img(input_img, background)
    input_img = input_img[int(0.15*height):int(0.85*height),:,:]
    cv2.imwrite("/home/gytang/project/dataset/yg_bak_0722/roll_rotation_padding_1/"+i, input_img)