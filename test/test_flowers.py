import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
sys.path.append('..')
# if you use tensorflow.keras:
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
# test image
test_b_2="../predict/pics/ROI.png"
image = imread(test_b_2)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

# loading pretrained model
model = EfficientNetB0(weights='imagenet')

# preprocess input
image_size = model.input_shape[1]
x = center_crop_and_resize(image, image_size=image_size)
x = preprocess_input(x)
print(x.shape)
x = np.expand_dims(x, 0)

# make prediction and decode
y = model.predict(x)
print(decode_predictions(y))