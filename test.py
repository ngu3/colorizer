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
        x = []
        y = []
        img_gray_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
        padding_img_gray_array = np.pad(img_gray_array, (1, 1), 'constant')
        img_color_carray = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_COLOR)
        img_gray_array = np.array(img_gray_array)
        print("shape,img_gray_array")
        print("shape,img_gray_array")
        print(img_gray_array.shape)
        print(img_gray_array)

        for j in range(0, img_gray_array.shape[1]):
            for i in range (0, img_gray_array.shape[0]):
                x.append(padding_img_gray_array[i:i+3 , j: j+3])
        x = np.array(x).reshape(img_gray_array.shape[0] * img_gray_array.shape[1], 3, 3, 1)
        # y = img_color_carray.transpose(1, 0, 2).reshape(-1, 1, 1, 3)
        y = img_color_carray.transpose(1, 0, 2).reshape(-1, 3)

        print("----------------rgbshape")
        # img_color_carray = np.array(img_color_carray)
        # print(img_color_carray.shape)

        if(len(input) == 0):
            input = x
        else:
            input = np.concatenate((input,x), axis=0)

        if(len(label) == 0):
            label = y
        else:
            label = np.concatenate((label,y), axis=0)
        # print("y.shape")
        print(img)
        print(y.shape)
        # for i in range(0,10):
        print(y[-3,0,0,:])
        # if y.all() == z.all():
        #     print("true")
        # print(y)
        # print(x.shape)
        # print(y.shape)
        # print(y)
        # print(y.shape)
    # print(len(input), len(label))
    # input = np.array(input)
    # # print(input.shape)
    # np.save('./data/ninput', input)
    # np.save('./data/nlabel', label)

    # np.savetxt('gray.csv', padding_img_gray_array.astype(int), fmt='%i', delimiter=',')
    # np.savetxt('labelr.csv', img_color_carray[:,:,0].astype(int), fmt='%i', delimiter=',')
    # np.savetxt('labelg.csv', img_color_carray[:, :, 1].astype(int), fmt='%i', delimiter=',')
    # np.savetxt('labelb.csv', img_color_carray[:, :, 2].astype(int), fmt='%i', delimiter=',')
    # print(input.shape)
    # print(label.shape)
    # for i in range(0,10):
    #
    # print(x)


create_traning_data()
