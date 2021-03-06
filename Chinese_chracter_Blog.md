## 方式一：迁移方式

​	只训练特征提取后的两层，EPOCH=10，BATCH_SIZE=16,训练结果为，keras默认学习率=0.001，此时的训练集和验证集上的loss和准确率如下：


使用迁移时，开放训练卷积层时，需要在base_model后紧接着进行，添加上新的层后，model会把base_model看成是一个层。在黑底白字上的效果如下：

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

![image-20200722100420454](./image-20200722100420454.png)

​	可以看到此时网络出现了**过拟合**的问题。

## 方式二：自建网络

没有使用很深的网络分类汉字(白底黑字),出现了不错的效果。模型结构如下：

```python
def build_net_003(input_shape, n_classes):
    model = tf.keras.Sequential([
        keras.layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model
```

思考：为什么有时很大很深的网络效果不好，反而不深较为简单的网络表现的却很不错呢？

​	可能是网络回退问题。



## 数据处理真的很重要！！！

解决了白底黑字的文字识别不行的问题：将转化的去掉

```python
image=tf.image.convert_image_dtype(image,tf.float32)
```

在使用tf.image.convert_image_dtype方法将一个uint类型的tensor转换为float类型时，该方法会自动对数据进行归一化处理，将数据缩放到0-1范围内，如果没有注意到这点，再除以255，之后在进行网络训练时会发现网络不收敛、不训练。

如果执行此语句时，image类型本来就行float类型，那么float32不会对图像进行归一化。

黑-->0

白-->255

而且在执行resize方法时，若image原来为int类型那么，会自动转成float类型。

```python
image = tf.image.resize(image, [64, 64])
```

所以黑底的照片在进行了reszie->covert->/255后依然能保留白色的特征，





## 彩色图片分类：

方案：使用迁移学习，迁移EfficiencientNetB4 只训练最后5层，设置学习率为0.001，batch_size=128,输入图像大小为64*64，epoch=100。训练结果为：

![image-20200728190513404](C:\Users\len\AppData\Roaming\Typora\typora-user-images\image-20200728190513404.png)

使用自己的数据集结果：

| 模型         | 结果                                                         | 测试集结果                                    | 数据集                        |
| ------------ | ------------------------------------------------------------ | --------------------------------------------- | ----------------------------- |
| net003       | ![image-20200818142629158](C:\Users\len\AppData\Roaming\Typora\typora-user-images\image-20200818142629158.png) | initial loss: 1.90<br>initial accuracy: 0.73  | 原始                          |
| efficientNet |                                                              | initial loss: 1.79<br/>initial accuracy: 0.63 | 原始                          |
| M5-HCCR      | ![image-20200818172840397](C:\Users\len\AppData\Roaming\Typora\typora-user-images\image-20200818172840397.png) | initial loss: 0.20<br/>initial accuracy: 0.99 | 只有训练集张数>500的类(100类) |
|              |                                                              |                                               |                               |

