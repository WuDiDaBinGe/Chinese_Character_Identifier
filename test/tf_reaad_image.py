import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
test="E://MyDocuments//1151680016//FileRecv//FileRecv_1//18.png"
test_w_b="../predict/pics/18.png"
test_b_2="../predict/pics/ROI.png"


image=tf.io.read_file(test_b_2)
image=tf.image.decode_png(image,channels=3)
print(image.shape)
#_, img_binary = cv2.threshold(image.numpy(), 200, 255, cv2.THRESH_BINARY)
#img_binary=np.expand_dims(img_binary,axis=-1)
#print(img_binary.shape)



# resize方法会将数据自动int转化为float类型
image = tf.image.resize(image, [256, 256])

image /= 255.0

# cv2.imshow("img", np.array(image.numpy()))
# cv2.waitKey(0)


plt.figure(1) # 图像显示
plt.imshow(image)
plt.show()


image_test=cv2.imread(test_b_2)
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
image_test = image_test - np.array(MEAN_RGB)
image_test = image_test / np.array(STDDEV_RGB)
print(image_test)