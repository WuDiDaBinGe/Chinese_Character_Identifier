import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.path.append("../")
import os
import tensorflow as tf
from pre_process import pics_dataset
import pickle
import cv2
TRAIN_PATH="/home/wbq/yuxiubin/data_dianxuan/data/train"
# 获取汉字label映射表
def get_label_dict(path):
    _,_,dict=pics_dataset.get_dataSet(path)
    return dict
# 将unicode编码转成汉字
def unicode_to_chinese(unicode_str):
    str_code = unicode_str[2:]
    str_code = '\\u' + str_code
    hanzi = str_code.encode('utf-8').decode("unicode_escape")
    return hanzi

def get_TOP_5(res,dict):
    # 从小到大排序 最大的概率在最后几位
    top_5_index = res.argsort(1)[0][-5:]
    res_result = []
    pro_result=[]
    top_5_index=list(reversed(top_5_index))
    for index in top_5_index:
        chinese=unicode_to_chinese(dict[index])
        res_result.append(chinese)
        pro_result.append(res[0][index])
    print(res_result)
    print(pro_result)
    return res_result

def get_Top_1(res,dict):
    top_index = res.argmax(1)[0]
    return dict[top_index]

def load_test_dataset(dataset_test_path):
    test_ds, test_num = pics_dataset.get_dataSet(dataset_test_path)
    return test_ds,test_num

def model_predict(pics_path, model_path):
    label_to_unicode_dict=get_label_dict(TRAIN_PATH)
    # key-value转换
    index_to_chara_dict = {index: name for name, index in label_to_unicode_dict.items()}
    # load model
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    # model.summary()

    # load image
    paths=os.listdir(pics_path)
    paths.sort()
    print(paths)
    for path in paths:
        full_path=os.path.join(pics_path,path)
        img_tensor=pics_dataset.load_and_preprocess_image(full_path)
        # 扩充一维增加 batch,h,w,channels
        img_tensor=tf.expand_dims(img_tensor,0)
        res=model.predict(img_tensor)
        print(path)
        get_TOP_5(res,index_to_chara_dict)


if __name__ == '__main__':
    dict=get_label_dict(TRAIN_PATH)

    model_predict('./predict/color_characters/','./model_save')