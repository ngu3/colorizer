import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = './data'

a = []
b = [[2,3,4],[2,3,4],[3,4,5]]
c = [[2,3,4],[2,3,5]]
a = a+b+c
print(a)

b=np.array(b).reshape(3,3,1)
print(b.shape)
print(b)


c=np.array(c).reshape(2,3,1)
print(c.shape)
print(c)

new = np.concatenate((a,c), axis = 0)
print(new.shape)
# # new = b+c
# # print(new)
#
# b = b + c
# print(b)