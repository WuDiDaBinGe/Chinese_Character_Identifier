import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

image=tf.io.read_file("../predict/pics/1572.png")
image=tf.image.decode_png(image,channels=3)



image = tf.image.resize(image, [128, 128])

image /= 255.0
plt.figure(1) # 图像显示
plt.imshow(image)
print(image.numpy().min())
print(image.numpy().max())


plt.show()