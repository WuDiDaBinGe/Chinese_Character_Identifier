import os
from PIL import Image

SRC_IMAGE_ROOT="F://dataset//Inspur_VrifyCode//15_below"
OUT_IMAGE_ROOT="F://dataset//Inspur_VrifyCode//15_below_guizheng"

def get_size_from_name(name_str):
    '''
    从图片名称字符串中获取图片大小 图片名称字符串：image1577454499__2__64.jpg
    :return: int 图片的大小
    '''
    name=name_str.split(".")[0]
    size=name.split("__")[-1]
    return int(size)

def ResizeImage(file_in,file_out,width,height):
    '''
    filein: 输入图片
    fileout: 输出图片
    width: 输出图片宽度
    height:输出图片高度
    '''
    img=Image.open(file_in)
    img_resize=img.resize((width,height),Image.ANTIALIAS)
    img_resize=img_resize.convert('RGB')
    img_resize.save(file_out)

if __name__ == '__main__':
    characters_list=os.listdir(SRC_IMAGE_ROOT)
    characters_list.sort()
    # 遍历汉字文件夹
    for charc in characters_list:
        characters_path=os.path.join(SRC_IMAGE_ROOT,charc)
        img_list=os.listdir(characters_path)
        img_list.sort()
        # 保存路径
        charc_out_dir = os.path.join(OUT_IMAGE_ROOT, charc)
        if not os.path.exists(charc_out_dir):
            os.mkdir(charc_out_dir)
        # 遍历文件夹中的图片
        for img_name in img_list:
            print(charc,img_name)
            img_full_path=os.path.join(characters_path,img_name)
            size=get_size_from_name(img_name)

            out_img_path=os.path.join(charc_out_dir,img_name)
            ResizeImage(img_full_path,out_img_path,size,size)
            print("规整成功....", size)






