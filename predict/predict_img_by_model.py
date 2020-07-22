import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import os
import tensorflow as tf
from pre_process import pics_dataset
import pickle
# 获取汉字label映射表
def get_label_dict():
    f=open('../chinese_labels','rb')
    character_dict = pickle.load(f)
    print(character_dict)
    f.close()
    return character_dict

def get_TOP_5(res,dict):
    # 从小到大排序 最大的概率在最后几位
    top_5_index = res.argsort(1)[0][-5:]
    res_result = []
    pro_result=[]
    top_5_index=list(reversed(top_5_index))
    for index in top_5_index:
        res_result.append(dict[index])
        pro_result.append(res[0][index])
    print(res_result)
    print(pro_result)
    return res_result

def get_Top_1(res,dict):
    top_index = res.argmax(1)[0]
    return dict[top_index]

def predict_with_model(pics_path,model_path):
    dict=get_label_dict()
    # load model
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    # model.summary()

    # load image
    paths=os.listdir(pics_path)
    for path in paths:
        full_path=os.path.join(pics_path,path)
        img_tensor=pics_dataset.load_and_preprocess_image(full_path)

        # 扩充一维增加 batch,h,w,channels
        img_tensor=tf.expand_dims(img_tensor,0)
        res=model.predict(img_tensor)
        print(path)
        get_TOP_5(res,dict)


if __name__ == '__main__':
    predict_with_model("./pics","../model_save")