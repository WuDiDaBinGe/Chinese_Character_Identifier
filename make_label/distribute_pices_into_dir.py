import pandas as pd
import shutil
import numpy as np
import os
import sys

LABEL_PATH=r"E://MyCoding//Jupyter//total_V2.csv"
IMG_ROOT_DIR="F://dataset//hanzi_dataset//pics_64//pics_64_64"
OUT_DIR="F://dseatat//hanzi_dataset//pics_64//dataset"


if __name__ == '__main__':
    df_label=pd.read_csv(LABEL_PATH)
    for index,row in df_label.iterrows():
        img_name=row['name']
        img_full_path=os.path.join(IMG_ROOT_DIR,img_name)
        img_character=row['value']
        out_dir_full_path=os.path.join(OUT_DIR,img_character)
        # 若没有则创建该文件夹
        if not os.path.exists(out_dir_full_path):
            os.mkdir(out_dir_full_path)
        # 复制图片
        shutil.copy(img_full_path,out_dir_full_path)
        print(row['name']," 移动完毕！")
    print("遍历完成！")