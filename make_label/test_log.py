import logging
import json
import pandas as pd
import numpy as np
import os
df=pd.read_csv('1.csv',encoding='utf-8')

for index,row in df.iterrows():
  print(row['name'], row['r_json'], type(row['name']), type(row['r_json']))
  # str转字典类型
  j = eval(row['r_json'])
  print(j['pic_str'])
  os.mkdir()



