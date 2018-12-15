import os
DATADIR = '../validation_data'
for img in os.listdir(DATADIR):
    print(img)
    img_gray_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
    # img_color_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_COLOR)
    padding_img_gray_array = np.pad(img_gray_array, (1, 1), 'constant')
    x = []
    # for j in range(0, img_gray_array.shape[1]):
    #     for i in range(0, img_gray_array.shape[0]):
    #         x.append(padding_img_gray_array[i:i + 3, j: j + 3]/25.5)
    for j in range(0, img_gray_array.shape[1]):
        for i in range(0, img_gray_array.shape[0]):
            x.append(padding_img_gray_array[i:i + 3, j: j + 3] / 255)

    x = np.array(x).reshape(img_gray_array.shape[0] * img_gray_array.shape[1], 3, 3, 1)
    res = model.predict(x)
    res = res * 255
    res = res.reshape(img_gray_array.shape[1], img_gray_array.shape[0], 3).transpose(1, 0, 2)
    cv2.imwrite(img[:2] + '_' + 'gray.jpg', res)