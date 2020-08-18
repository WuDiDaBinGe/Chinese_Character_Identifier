from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import os
import pathlib
import random
import cv2
import numpy as np

DATASET_ROOT_PATH="/home/wbq/yuxiubin/dataset_yes_15below_own_unicode/dataset_above_500/train"

def binary_image(img):
    img=img.numpy()
    mask=img>=200
    img[mask]=255
    return img

def preprocess_image(image,img_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    # 对数据进行归一化，将image转到（0，1）的范围内
    #image=tf.image.convert_image_dtype(image,tf.float32)

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
    path=pathlib.Path(path)
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
    return all_image_paths,all_image_labels,label_to_index

def create_DataSet(all_img_paths,all_img_labels):
    data_count=len(all_img_paths)
    print(data_count)
    path_ds=tf.data.Dataset.from_tensor_slices(all_img_paths)
    image_ds=path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
    label_ds=tf.data.Dataset.from_tensor_slices(tf.cast(all_img_labels,tf.int64))
    image_label_ds=tf.data.Dataset.zip((image_ds,label_ds))
    return image_label_ds,data_count

def set_batch_shuffle(batch_size,ds,count):
    # 使用缓存-加快读取性能
    img_ds = ds.cache(filename='./cache.tf-data')
    # 将数据集完全打乱
    img_ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=count//2))
    # 设置batch 方便使用prefetch读取
    img_ds=img_ds.batch(batch_size)
    img_ds=img_ds.prefetch(1)
    return img_ds

def change_range(image,label):
    return 2*image-1,label

def get_dataSet(path):
    all_image_path,all_labels,label_name_dict=read_imgs_path_labels(path)
    ds, count = create_DataSet(all_image_path, all_labels)
    return ds,count,label_name_dict

if __name__ == '__main__':
    ds,count,dict=get_dataSet(DATASET_ROOT_PATH)
    print(dict)
    ds=set_batch_shuffle(32,ds,count)
    for image_batch, label_batch in ds.take(1):
        pass
    print(image_batch.shape)
