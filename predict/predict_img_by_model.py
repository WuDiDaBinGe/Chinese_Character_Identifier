import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pre_process import pics_dataset
import pickle
# 获取汉字label映射表
def get_label_dict():
    f=open('../chinese_labels','rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

def predict_with_model(pics_path,model_path):
    dict=get_label_dict()
    # load model
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    #model.summary()

    # load image
    paths=os.listdir(pics_path)
    for path in paths:
        full_path=os.path.join(pics_path,path)
        img_tensor=pics_dataset.load_and_preprocess_image(full_path)
        img_tensor=2*img_tensor-1
        # 扩充一维增加batch,h,w,channels
        img_tensor=tf.expand_dims(img_tensor,0)
        res=model.predict(img_tensor)
        top_5_index=res.argsort(1)[0][:5]
        print(float(res[0][top_5_index[0]]))
        print(path)
        res_result=[]
        for index in top_5_index:
            res_result.append(dict[index])
        print(res_result)
if __name__ == '__main__':
    predict_with_model("./pics","../model_save")