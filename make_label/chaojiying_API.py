import sys
import requests
from hashlib import md5
import os
import logging
import pandas as pd
file_handler = logging.FileHandler(filename='x1.log', mode='a', encoding='utf-8',)

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S %p',
  handlers=[file_handler,],
  level=logging.DEBUG
)

class Chaojiying_Client(object):

    def __init__(self, username, password, soft_id):
        self.username = username
        password = password.encode('utf8')
        self.password = md5(password).hexdigest()
        self.soft_id = soft_id
        self.base_params = {
            'user': self.username,
            'pass2': self.password,
            'softid': self.soft_id,
        }
        self.headers = {
            'Connection': 'Keep-Alive',
            'User-Agent': 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0)',
        }

    def PostPic(self, im, codetype):
        """
        im: 图片字节
        codetype: 题目类型 参考 http://www.chaojiying.com/price.html
        """
        params = {
            'codetype': codetype,
        }
        params.update(self.base_params)
        files = {'userfile': ('ccc.jpg', im)}
        r = requests.post('http://upload.chaojiying.net/Upload/Processing.php', data=params, files=files, headers=self.headers)
        return r.json()

    def ReportError(self, im_id):
        """
        im_id:报错题目的图片ID
        """
        params = {
            'id': im_id,
        }
        params.update(self.base_params)
        r = requests.post('http://upload.chaojiying.net/Upload/ReportError.php', data=params, headers=self.headers)
        return r.json()


if __name__ == '__main__':
    len_args=len(sys.argv)
    chaojiying = Chaojiying_Client('ecdgcs', 'drmfslx', '906932')	#用户中心>>软件ID 生成一个替换 96001
    pics_dir_path='test_image'
    pics_paths=os.listdir(pics_dir_path)
    pics_paths.sort()
    if len_args==2:
        split_image_name=sys.argv[1]
        split_index=pics_paths.index(split_image_name)
        pics_paths=pics_paths[split_index+1:split_index+2]
    else:
        pass
    r_json_list=[]
    read_paths=[]
    for path in pics_paths:
        full_image_path=os.path.join(pics_dir_path,path)
        im = open(full_image_path, 'rb').read()	    # 本地图片文件路径 来替换 a.jpg 有时WIN系统须要//

        r_json=chaojiying.PostPic(im, 2001)         # 1902 验证码类型  官方网站>>价格体系 3.4+版 print 后要加()
        print(type(r_json))
        # 判断是否正确
        # 打印并写入日志文件
        print(path, ",", r_json)
        msg=path+","+str(r_json)
        logging.info(msg)

        # 写入csv文件
        read_paths.append(path)
        r_json_list.append(r_json)
        dit = {'name': read_paths, 'r_json': r_json_list}
        df = pd.DataFrame(dit)
        # 追加csv文件
        df.to_csv(r'./1.csv', index=False, sep=',', encoding='utf-8',mode='a',header=False)
        r_json_list.clear()
        read_paths.clear()

