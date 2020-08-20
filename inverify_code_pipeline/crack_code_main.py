from inverify_code_pipeline import jieba_wordsequence_API
from inverify_code_pipeline import search_wordsequence
from inverify_code_pipeline import detect_character
from predict import predict_colorimg_bymodel
import tensorflow as tf
import os
from functools import reduce
import json
def load_label_unicode_dict(dict_path):
    '''
    :param dict_path:加载json文件的路径
    :return: 返回字典（key-value互换后）: {label:uncodestring}
    '''
    with open(dict_path) as f:
        dict_label_unicode=json.load(f)
    # key-value转换
    index_to_chara_dict = {index: name for name, index in dict_label_unicode.items()}
    return index_to_chara_dict

# 求多个列表的组合
def combination(*lists):
    total = reduce(lambda x, y: x * y, map(len, lists))
    retList = []
    for i in range(0, total):
        step = total
        tempItem = []
        for l in lists:
            step /= len(l)
            tempItem.append(l[int(i/step % len(l))])
        retList.append(tuple(tempItem))
    return retList

def crack_verification_code(verification_path,detectmodel_path,distinguishmodel_path):
    dir_path,img_name=os.path.split(verification_path)
    img_name=img_name.split(".")[0]
    # 加载分类的标签对应的Unicode字符串字典
    label_uncicode_dict=load_label_unicode_dict("./unicode-label.json")
    # 使用检测模型检测汉字
    character_rects,character_save_names=detect_character.detect_character(verification_path,detectmodel_path)
    print(character_save_names)

    result_list=[]
    # 汉字与位置的坐标的对应字典
    hanzi_position_dict={}
    # 遍历切割的汉字，进行分类
    index=0
    for charac_img_path in character_save_names:
        top_5_result,top_5_pro=predict_colorimg_bymodel.predict_single_img(charac_img_path,distinguishmodel_path,label_uncicode_dict)
        if top_5_pro[0]>0.97:
            tmp_list=[charc for charc in top_5_result[:1]]
        else:
            tmp_list=[charc for charc in top_5_result]
        for tmp_character in tmp_list:
            hanzi_position_dict[tmp_character]=character_rects[index]
        index+=1
        result_list.append(tmp_list)
    print(hanzi_position_dict)
    combination_hanzi=combination(*result_list)
    print(combination_hanzi)
    combination_connect_hanzi=[]
    #  将字符连接起来
    for words in combination_hanzi:
        combination_connect_hanzi.append(''.join(words))

    jieba_flag=0
    #  识别语序
    for words in combination_connect_hanzi:
        rec_word_possible = jieba_wordsequence_API.recog_order_jieba(words)
        if rec_word_possible:
            jieba_flag = 1
            break
    # 结巴分词识别出来了 就确定结果
    # 若没识别出来 选择最高置信度的单词组合 用搜索引擎进行识别
    if jieba_flag:
        res_word=rec_word_possible
    else:
        rec_word_possible=search_wordsequence.search_engine_recog(combination_connect_hanzi[0])
        res_word=rec_word_possible

    # 根据汉字位置字典，选出最终的汉字对应的位置
    res_center_posiotion=[]
    for chara in res_word:
        xmin,xmax,ymin,ymax=hanzi_position_dict[chara]
        center_x=(xmin+xmax)/2
        center_y=(ymin+ymax)/2
        res_center_posiotion.append((center_x,center_y))
    return res_center_posiotion,res_word



if __name__ == '__main__':
    posis,res_list=crack_verification_code("./t_images/verifyCode1576132228.jpg","../detect_models/saved_model_new","../character_classfy_model/model_save_20200817_M5_allclss")
    print(posis)
    print(res_list)


