import cv2
import shutil
import os
import sys
'''
    使用说明：
    一、
    python showimgs_incharacter_tools.py ./test_image(图像文件夹) ./(图像以及创建的文件夹目录)
    二、
    是否从中间某张图片开始（no：从头开始遍历文件夹，imgXXX.jpg从该图片开始遍历）：
    1.输入no 表示从头遍历图像文件夹
    2.输入image1.jpg(该图像要在输入图像文件中存在) 从该图片开始遍历文件夹中的图像
    三、
    Please input the value in characters3.jpg 
    输入图像显示的汉字
'''
def show_img(img_path):
    im = cv2.imread(img_path)
    cv2.imshow(img_path, im)
    key = cv2.waitKey(0)
    cv2.destroyWindow(img_path)

if __name__ == '__main__':
    input_dir=sys.argv[1]
    output_dir=sys.argv[2]
    imgs = os.listdir(input_dir)
    imgs.sort()
    while True:
        choice=input("是否从中间某张图片开始（no：从头开始遍历文件夹，imgXXX.jpg从该图片开始遍历）：")
        print(choice)
        list_no=['no','NO','No']
        if choice in list_no:
            print("从头遍历",input_dir)
            break
        elif choice in imgs:
            img_index = imgs.index(choice)
            imgs = imgs[img_index + 1:]
            break
        else:
            print("输入错误，检查输入图片名称！")

    for img_path in imgs:
        full_img_path=os.path.join(input_dir,img_path)
        # 读取并显示图片
        show_img(full_img_path)
        while True:
            # 输入的图片的值
            value = input("Please input the value in {}:".format(img_path))
            if len(value)==1:
                break
            else:
                print("Input error！Please input again！")
        target_dir=os.path.join(output_dir,value)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        # 复制图片
        shutil.copy(full_img_path,target_dir)
    print("本次共标记{}幅图片".format(len(imgs)))




