import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def create_traning_data():
    DATADIR = './data'
    input = []
    label = []
    for img in os.listdir(DATADIR):
        if '.jpg' not in img:
            continue
        print(img)
        x = []
        y = []

        # read gray picture and padding 0s
        img_gray_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
        padding_img_gray_array = np.pad(img_gray_array, (3, 3), 'constant')
        # print(img_gray_array.shape)
        # print("shape of padding array")
        # print(padding_img_gray_array.shape)
        # print(padding_img_gray_array)
        # read color picture and prepared for labels
        img_color_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_COLOR)
        # i = number of rows, j = number of columns, sliding slice 3*3 pixels in a picture
        # img_gray_test = img_gray_array.reshape(-1, 1200, 1600, 1)
        for j in range(0, img_gray_array.shape[1]):
            for i in range (0, img_gray_array.shape[0]):
                x.append(padding_img_gray_array[i:i+7 , j: j+7]/255)

        # convert inputs to 4 dims
        x = np.array(x).reshape(img_gray_array.shape[0] * img_gray_array.shape[1], 7, 7, 1)

        y = img_color_array.transpose(1, 0, 2).reshape(-1, 1, 1, 3)/255
        # y = img_color_array.transpose(1, 0, 2).reshape(-1, 3)
        # concatenating the each element of input and label
        if(len(input) == 0):
            input = x
        else:
            input = np.concatenate((input,x), axis=0)

        if(len(label) == 0):
            label = y
        else:
            label = np.concatenate((label,y), axis=0)


    # -------------------------------------------------------
    # save inputs and labels to file
    np.save('./data/n1_77_input', input)
    np.save('./data/n1_77_label', label)
    # -------------------------------------------------------

    # tidious testing
    # x = np.concatenate((x,x), axis=0)
    # print(x.shape)
    # print('---------------')
    # print(x[0, :])
    # print('---------------')
    # print(x[1920000,:])
    # for i in range(0,1920000):
    #     # if(i == 1920000-1):
    #     #     print("true")
    #     if x[0, :].all() != x[1920000, :].all():
    #         print("wrong")
    # print("0")
    # print(img_color_array.shape)
    # print(img_color_array[:, :, 0].shape)
    # print('---------------')
    # print(img_color_array[:, :, 0])
    # print('---------------')
    # print(img_color_array[:, :, 1])
    # print('---------------')
    # print(img_color_array[:, :, 2])
    # testy = img_color_array.transpose(1, 0, 2).reshape(-1, 1, 1, 3)
    # print(testy.shape)
    # print(testy[0:10:,:,:])

    # restore testing
    # print(label.shape)
    # res = label.reshape(img_gray_array.shape[1], img_gray_array.shape[0], 3).transpose(1,0,2)
    # print(res.shape)
    # cv2.imwrite('test.jpg', res)


create_traning_data()
