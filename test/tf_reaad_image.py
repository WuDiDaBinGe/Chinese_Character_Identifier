import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
test="E://MyDocuments//1151680016//FileRecv//FileRecv_1//18.png"
test_w_b="../predict/pics/18.png"
test_b_2="../predict/pics/5.png"
image=tf.io.read_file(test)

image=tf.image.decode_png(image,channels=1)
print(image.shape)
_, img_binary = cv2.threshold(image.numpy(), 200, 255, cv2.THRESH_BINARY)
img_binary=np.expand_dims(img_binary,axis=-1)
print(img_binary.shape)



# resize方法会将数据自动int转化为float类型
image_64 = tf.image.resize(img_binary, [64, 64])

image_64 /= 255.0

cv2.imshow("img", np.array(image_64.numpy()))
cv2.waitKey(0)


# plt.figure(1) # 图像显示
# plt.imshow(image)
# plt.show()