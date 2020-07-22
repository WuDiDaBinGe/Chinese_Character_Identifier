迁移方式：

1. 方式一：只训练特征提取后的两层，EPOCH=10，BATCH_SIZE=16,训练结果为，keras默认学习率=0.001，此时的训练集和验证集上的loss和准确率如下：

   

   

2. 



使用迁移时，开放训练卷积层时，需要在base_model后紧接着进行，添加上新的层后，model会把base_model看成是一个层。





pb模型测试结果如下：

14.png
['津\r', '荤\r', '晦\r', '晤\r', '晕\r']
21.png
['津\r', '碘\r', '碟\r', '碧\r', '碰\r']
33.png
['津\r', '矽\r', '矾\r', '矿\r', '码\r']
34.png
['津\r', '舟\r', '是\r', '舞\r', '显\r']
46.png
['津\r', '碎\r', '碑\r', '碗\r', '碘\r']
59.png
['津\r', '碗\r', '碘\r', '碟\r', '碧\r'

![image-20200722100420454](C:\Users\len\AppData\Roaming\Typora\typora-user-images\image-20200722100420454.png)