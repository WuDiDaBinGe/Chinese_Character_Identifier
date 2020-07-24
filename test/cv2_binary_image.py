import os
import sys
import cv2

def binary_images(in_path,out_path):
    if not os.path.exists(in_path):
        return
    paths=os.listdir(in_path)
    num=0
    for path in paths:
        full_path=os.path.join(in_path,path)
        image=cv2.imread(full_path)

        _,binary_image=cv2.threshold(image,220,255,cv2.THRESH_BINARY)
        cv2.imwrite(out_path+"//"+str(num)+".png", binary_image)
        num += 1
        cv2.imshow("img"+str(num),binary_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    binary_images("E://MyDocuments//1151680016//FileRecv//FileRecv_1","E://MyDocuments//1151680016//FileRecv//FileRecv_2")