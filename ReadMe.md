## 验证码识别

![](.\crack_inverifycode_pipeline\t_images\image2.jpg)

使用深度模型技术识别验证码中的汉字。

深度学习技术解决两方面的需求：

1. 从图片中检测出汉字的位置--训练一个检测模型
2. 创建汉字数据集，训练汉字识别模型。

在两个模型的作用下，最终结合结巴分词库和搜索引擎库调整语序得到验证码中词语的正确语序。

##### Requirements



##### Chinese_Character_Identifier目录

- `pre_process`包：保存了数据集读入以及数据预处理的代码
- `predict`包：保存了使用模型预测的文件
- `train`包：包含了训练代码
- `test`包：保存了一些测试文件
- `inverify_code_pipeline`:验证码破解的完整流程，封装了jieba分词和搜索引擎等API
- `detect_models`包：存放检测模型的包
- 

