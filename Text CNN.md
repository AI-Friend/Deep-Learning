# Text CNN综述及实现

Li Junli 李军利  /  June 7th 2019



## 1. Text CNN综述

**TextCNN**是CNN的一个变种，CNN主要运用于图片分类，而TextCNN主要用于文本分类。2014年**Yoon Kim**在论文提出用于文本分类的TextCNN方法 [1] —— 将卷积神经网络 应用到文本分类任务，利用多个**不同size**的**kernel**来提取句子中的关键信息（类似于多窗口大小的**n-gram**），从而能够更好地捕捉局部相关性。

大多数NLP任务的输入是表示为矩阵的句子或文档。矩阵的每一行对应一个标记，通常是一个单词（但它可以是一个字符，甚至一个句子）。也就是说，每行是表示单词的向量。通常，这些向量是 **word embedding** ，如[word2vec](https://code.google.com/p/word2vec/)或[GloVe](http://nlp.stanford.edu/projects/glove/)，但它们也可以是将单词索引为词汇表的**独热向量**。对于使用100维词向量的10个单词 组成的句子，我们将使用10×100矩阵作为输入。

处理图像时，过滤器是左右、上下左右滑动的，从而来提取特征值，最终获得特征矩阵，但是文本的话稍有不同，不能左右滑动，只能**上下滑动**，原因很简单，不能将一个单词分开来进行训练，如果非要这样的话，卷积之后获得的数据将会没有什么意义，因此，我们的滤波器的“宽度”通常与输入矩阵的宽度相同。

<center> Text CNN 的详细过程</center>
![](./media/Text CNN 过程.jpg)

**上图的详细过程**

```
Embedding：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点

Convolution：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel

MaxPolling：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示

FullConnection and Softmax：最后接一层全连接的 softmax 层，输出每个类别的概率。
```



**通道（Channels）**

```
图像中可以利用 (R, G, B) 作为不同channel

文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。（上图中的输入只有一个通道）
```

 

**一维卷积（conv-1d）**

```
图像是二维数据

文本是一维数据，因此在TextCNN卷积用的是一维卷积（在word-level上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要**通过设计不同kernel_size 的 filter 获取不同宽度的视野
```

 

**Pooling层**：

```
利用CNN解决文本分类问题的文章非常多，也有很多小的tricks，比如 nal kalchbrenner[2] 等人的论文，文中将max pooling 改成 (dynamic) k-max pooling ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息

举个情感分析场景的例子，对于句子  "我觉得这个地方景色还不错，但是人也实在太多了"

虽然前半部分体现的情感是正向的，全局文本表达的却是**偏负面**的情感，利用 **k-max pooling** 能够很好捕捉这类信息
```



**Word Embedding**

```
数据量较大：可以直接随机初始化embeddings，然后基于语料通过训练模型网络来对embeddings进行更新和学习

数据量较小：可以利用外部语料来预训练(pre-train)词向量，然后输入到Embedding层，用预训练的词向量矩阵初始
          化embeddings。（通过设置weights=[embedding_matrix]）

          静态(static)方式：训练过程中不再更新embeddings。实质上属于迁移学习，特别是在目标领域数据量比
                          较小的情况下，采用静态的词向量效果也不错。（通过设置trainable=False）

          非静态(non-static)方式：在训练过程中对embeddings进行更新和微调(fine tune)，能加速收敛。
                               （通过设置trainable=True）
  
注：参数用的Keras框架下的名称和设置
```



## 2. 实现(TensorFlow)

上一章对 **Text CNN** 作了综述，着重介绍了其与经典CNN的区别，其余大部分与经典CNN相似。经典CNN算法可以参考个人 [GitHub](https://github.com/Ljl-Jdsk/) 上的另一篇文章 [CNN全解](https://github.com/Ljl-Jdsk/AI-DeepLearning/blob/master/CNN%E5%85%A8%E8%A7%A3.pdf) 。

本章用TensorFlow框架，实现一个文本分类的例子，在github上找到了现成的数据和python Package[4]，如果读者想要自己实现，可以从[4]中下载数据，本章只把自己本地调通的核心代码展示出来。



### 2.1 数据集

本次训练使用了THUCNews的一个**10分类**的新闻子集。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

这个子集可以在此下载：链接: <https://pan.baidu.com/s/1hugrfRu> 密码: qfud

数据集划分如下：

- cnews.train.txt: 训练集(5000*10)
- cnews.val.txt: 验证集(500*10)
- cnews.test.txt: 测试集(1000*10)

以验证集为例，10个类别，每个类别 500条，每行一条数据，标签在前，内容在后，每条新闻很长，...作省略

```
体育	黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛首轮...
体育	杜兰特41分奇兵送9盖帽 雷霆4-1掘金晋级第二轮新浪体育讯北京时间4月28日，雷霆主场最后...
娱乐	北美票房综述：《速度与激情4》王牌归位(图)本周综述2009年的第一个“首映性话题”正式诞生！...
家居	橱柜价格关键在计价方式 教你如何挑选买过橱柜的人都知道，橱柜的计价很复杂，商家的报价方式...
房产	北京玫瑰园 因不同而至尚北京玫瑰园 (论坛 相册 户型 样板间 地图搜索)，一个京城地标性...
教育	从2010年6月英语四级考试谈12月复习备考来源：文都教育 2010年6月份大学英语四级考试的脚步...
...............................................................................
```

*作为一个篮球迷，看到科比，杜兰特，emmm，这两条应是2011年季后赛首轮湖人VS黄蜂，雷霆VS掘金的新闻。现如今已经到了2019年的NBA总决赛了，物是人非，3年前，科比退役，KD背负骂名远走金州。现在总决赛舞台上站着的是意欲3连冠的勇士和第一次打进总决赛的猛龙，勇士队饱受伤病困扰，尤其KD在与西部半决赛第五战受伤后即高挂免战牌，复出困难，也导致勇士目前大比分1:3落后于猛龙。一年前饱受争议的卡哇伊被马刺交易到猛龙，现在很大概率是他封神的一年。而金州勇士，不管KD是否复出，如果想洗刷当年1:3被逆转的耻辱，那就去逆转别人吧！！！在学习之余插一段篮球日记，突然感觉我有当体育小编的潜质！！！*



### 2.2 数据预处理

本文采用字符级的Text CNN，需要建立字符词典；设置固定的每一条新闻文本的序列长度（字数）为600、词向量的长度为64。

`Data/data_processing.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data    | Shape        | Data    | Shape       |
| ------- | ------------ | ------- | ----------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val   | [5000, 600]  | y_val   | [5000, 10]  |
| x_test  | [10000, 600] | y_test  | [10000, 10] |





TextCNN类搭建了一个最basic的CNN模型，有input layer，convolutional layer，max-pooling layer和最后输出的softmax layer.
但是又因为整个模型是用于文本的(而非CNN的传统处理对象：图像)，因此在cnn的操作上相对应地做了一些小调整：

    对于文本任务，输入层自然使用了word embedding来做input data representation
    接下来是卷积层，大家在图像处理中经常看到的卷积核都是正方形的，比如4*4，然后在整张image上沿宽和高逐步移动进行卷积操作。但是nlp中输入的”image”是一个词矩阵，比如n个words，每个word用200维的vector表示的话，这个”image”就是n*200的矩阵，卷积核只在高度上已经滑动，在宽度上和word vector的维度一致（=200），也就是说每次窗口滑动过的位置都是完整的单词，不会将几个单词的一部分”vector”进行卷积，这也保证了word作为语言中最小粒度的合理性。（当然，如果研究的粒度是character-level而不是word-level，需要另外的方式处理）
    由于卷积核和word embedding的宽度一致，一个卷积核对于一个sentence，卷积后得到的结果是一个vector， shape=（sentence_len - filter_window + 1, 1），那么，在max-pooling后得到的就是一个Scalar.所以，这点也是和图像卷积的不同之处，需要注意一下
    正是由于max-pooling后只是得到一个scalar，在nlp中，会实施多个filter_window_size（比如3,4,5个words的宽度分别作为卷积的窗口大小），每个window_size又有num_filters个（比如64个）卷积核。一个卷积核得到的只是一个scalar太孤单了，智慧的人们就将相同window_size卷积出来的num_filter个scalar组合在一起，组成这个window_size下的feature_vector
    最后再将所有window_size下的feature_vector也组合成一个single vector，作为最后一层softmax的输入
    一个卷积核对于一个句子，convolution后得到的是一个vector；max-pooling后，得到的是一个scalar
————————————————
版权声明：本文为CSDN博主「M先森」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_33427047/article/details/80393972





## 声明







## References

[1]  Yoon Kim.Convolutional Neural Networks for Sentence Classification.EMNLP, 2014

[2]  nal kalchbrenner, edward grefenstette, phil blunsom.A Convolutional Neural Network for Modelling 

​       Sentences.Department of Computer Science University of Oxford

[3]  [CNN-RNN中文文本分类](http://www.tensorflownews.com/2017/11/04/text-classification-with-cnn-and-rnn/)

[4]  [CNN-RNN分类 github](https://github.com/gaussic/text-classification-cnn-rnn)

[5]  [Text CNN模型原理及实现](https://www.cnblogs.com/bymo/p/9675654.html)

