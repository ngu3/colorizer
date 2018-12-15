import h5py
from keras.models import Sequential
import numpy as np
from keras.models import load_model
from keras.layers import Dense,Conv2D,InputLayer, UpSampling2D
from keras.optimizers import SGD,RMSprop,Adam
import os
from numpy.random import seed
import cv2
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#input_dim 9
#output_dim 3

# # 3 hidden layer
model = Sequential()
model.add(Dense(units=6, input_dim=(9), activation='relu'))  # batch_size * 9 --> batch_size * 3 (RGB)
model.add(Dense(units=3, input_dim=(6), activation='relu'))  # batch_size * 9 --> batch_size * 3 (RGB)
# model.add(Dense(units=27, input_dim=(27), activation='relu'))
# model.add(Dense(units=27, input_dim=(9), activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mse')
# # loss: 0.0095 - val_loss: 0.0074
# loss: 0.0113 - val_loss: 0.0072 15 epoch
# 0.0106 - val_loss: 0.0071 epoch 5
# 28s 1us/step - loss: 0.0103 - val_loss: 0.0073 epoch10 batch size 1500
# 27s 1us/step - loss: 0.0103 - val_loss: 0.0073 epoch20
# model = Sequential()
# model.add(Dense(units=27, input_dim=(9), activation='relu'))  # batch_size * 9 --> batch_size * 3 (RGB)
# model.add(Dense(units=81, input_dim=(27), activation='relu'))
# model.add(Dense(units=81, input_dim=(81), activation='relu'))
# model.add(Dense(units=27, input_dim=(81), activation='relu'))
# model.add(Dense(units=27, input_dim=(9), activation='relu'))
# model.add(Dense(units=3, activation='relu'))
# model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mse')

# x_train = np.load('../data/n1_input.npy').astype(float)
x_train = np.load('../data/n1_input.npy')
x_train = x_train.reshape(-1,9)
y_train = np.load('../data/n1_label.npy')
y_train = y_train.reshape(-1,3)
x_test = np.load('../data/n1_validation_input.npy')
x_test = x_test.reshape(-1,9)
y_test = np.load('../data/n1_valication_label.npy')
y_test = y_test.reshape(-1,3)
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
print(x_train.shape)
# y_train = np.load()
model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test), batch_size=1500)

# test_on_validation
DATADIR = '../validation_data'
for img in os.listdir(DATADIR):
    if "predict" in img:
        continue
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
    x = x.reshape(-1,9)
    res = model.predict(x)
    res = res.reshape(-1,1,1,3)
    res = res * 255
    res = res.reshape(img_gray_array.shape[1], img_gray_array.shape[0], 3).transpose(1, 0, 2)

    cv2.imwrite(DATADIR+'/'+img[:2] + '_' + 'predict_fd1l_e8.jpg', res)
# model.train_on_batch(x_batch, y_batch)
model.save('my_base_model_fd1l_e8.h5')

# scores = model.evaluate(x_train, y_train)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# print(scores)z


