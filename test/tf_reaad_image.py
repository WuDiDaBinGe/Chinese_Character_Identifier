import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

image=tf.io.read_file("../predict/pics/1572.png")
image=tf.image.decode_png(image,channels=3)
image=tf.image.convert_image_dtype(image,tf.float32)
print(image.numpy().max())
# resize方法会将数据自动int转化为float类型
image = tf.image.resize(image, [64, 64])
print(image)
print(image.numpy().max())

image /= 255.0
print(image.numpy().max())

plt.figure(1) # 图像显示
plt.imshow(image)
plt.show()