from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import matplotlib.pyplot as plt
import os
import pathlib
import random
DATASET_ROOT_PATH="F://dataset//hanzi_dataset//dataset_character//dataset"
TRAIN_PATH=DATASET_ROOT_PATH+"//train"
TRAIN_PATH=pathlib.Path(TRAIN_PATH)
TEST_PATH=DATASET_ROOT_PATH+"//test"
TEST_PATH=pathlib.Path(TEST_PATH)
def preprocess_image(image,img_size):
  image = tf.image.decode_png(image, channels=3)
  image = tf.image.resize(image, [img_size, img_size])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image,img_size=64)

def read_imgs_path_labels(path):
    '''
    本函数读入一个路径，一句路径读取所有图片，并得到图片的类别变迁
    :param path:图片文件夹路径
    :return:图像路径，图像对应的标签
    '''
    # 得到所有图片的路径
    all_image_paths=list(path.glob('*/*'))
    all_image_paths=[str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    # 得到图片标签的名称
    label_names=sorted(item.name for item in path.glob('*/') if item.is_dir())
    # 为标签分配索引
    label_to_index=dict((name,index) for index,name in enumerate(label_names))
    # 得到图片的标签索引
    all_image_labels=[label_to_index[pathlib.Path(path_).parent.name] for path_ in all_image_paths]
    return all_image_paths,all_image_labels

def create_DataSet(all_img_paths,all_img_labels):
    data_count=len(all_img_paths)
    print(data_count)
    path_ds=tf.data.Dataset.from_tensor_slices(all_img_paths)
    image_ds=path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
    label_ds=tf.data.Dataset.from_tensor_slices(tf.cast(all_img_labels,tf.int32))
    image_label_ds=tf.data.Dataset.zip((image_ds,label_ds))
    return image_label_ds,data_count

def set_batch_shuffle(batch_size,ds,count):
    # 使用缓存-加快读取性能
    img_ds = ds.cache(filename='./cache.tf-data')
    # 将数据集完全打乱
    img_ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=count//3))
    # 设置batch 方便使用prefetch读取
    img_ds=img_ds.batch(batch_size)
    img_ds=img_ds.prefetch(1)
    return img_ds

def change_range(image,label):
    return 2*image-1,label

if __name__ == '__main__':
    all_image,all_labels=read_imgs_path_labels(TRAIN_PATH)
    ds,count=create_DataSet(all_image,all_labels)
    ds=set_batch_shuffle(16,ds,count)
    keras_ds=ds.map(change_range)
    mobile_net=tf.keras.applications.MobileNetV2(input_shape=(64,64,3),include_top=False)
    mobile_net.trainable=False
    image_batch,label_batch=next(iter(keras_ds))
    feature_map_batch=mobile_net(image_batch)
    print(feature_map_batch.shape)
