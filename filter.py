import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./validation_data/30_predict_fd1l_e20.jpg')

# # dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
dst = cv2.fastNlMeansDenoisingColored(img,None,3,3,3,7)
#
# plt.subplot(121),plt.imshow(img)
plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.subplot(122),plt.imshow(dst)
plt.show()