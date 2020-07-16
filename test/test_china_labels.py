import pickle
# 获取汉字label映射表
def get_label_dict():
    f=open('../chinese_labels','rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

label_dict=get_label_dict()
print(label_dict)
