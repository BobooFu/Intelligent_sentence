##大连市第一届大数据比赛限选题1（大数据智能量刑）
##对对对对对队

## 基于python3.5

## 所用库
numpy==1.14.2
jieba==0.39
pandas==0.22.0
tensorflow==1.8.0
Keras==2.1.6
scikit-learn==0.19.1

##正确打开方式##
Intelligent_sentence为工作目录，
在工作目录下run   /data_utils/data_preparation.py进行数据预处理步骤
在工作目录下run   train_textcnn_keras.py       训练模型
在工作目录下run   predictor.py                 输出结果

按照比赛要求的格式：
提交文件应为csv文件，首行为csv文件标题，使用英文逗号分隔，之后每行为一条案情事实预测的指控信息和法律条文结果。多条结果使用英文封号“;”进行分隔。
再对输出的"final_result.csv"的格式进行调整。
形成fianl.csv生成提交文件


## 任务介绍
赛题任务为嫌疑人罪名预测。
根据案情事实，针对嫌疑人罪名进行预测分类，
本次任务选取案件类型较多的TOP30作为罪名空间。
选手需要对罪名进行预测，每个案情事实可能对应多项罪名。评价指标使用宏平均的F1与微平均F1值加权的方式给出


## 数据
案件案情事实数据分为训练集和测试集两部分，格式为csv格式。其中以训练集数据为例，各个字段含义如下：

##字段及含义

ids：案件id信息，由案件正文内容计算hash得到
fact：案件对应的案情事实
criminal：需要进行罪名预测和法律条文预测的目标嫌疑人
accusation：指控信息，即嫌疑人涉及的罪名信息
articles：嫌疑人涉及到的法律条文编号

格式均为str


例子：
* **ids**:'9a8f8c3c61975645fdc27cce45cf839b'
* **fact**:'苏州市虎丘区人民检察院指控：2014年6月28日晚，被告人李某某在本市虎丘区何山路松园宿舍大门西侧摆摊时，因其妻子彭某某、儿子李某与尚某某（均行政处罚）为琐事发生纠纷继而扭打，在尚某某丈夫孙某某持斧子将李某砍伤后，被告人李某某遂持西瓜刀对被害人孙某某实施追砍，造成被害人孙某某头部、脸部及手臂不同程度损伤。经鉴定，被害人孙某某的损伤构成人体轻伤二‘级。公诉机关认为，被告人李某某的行为已构成××，鉴于其归案后如实供述自己的罪行，可以从轻处罚。为证实上述指控的事实及公诉意见，公诉人当庭讯问了被告人李某某，并宣读和出示了相关的证据材料。'
* **criminals**:'李某某'
* **accusation**:'故意伤害'
* **articles**:'234'

除案情数据外，针对法律条文数据给出编号对应的法律条文，条文根据《中华人民共和国刑法》整理，格式为csv格式，分隔符为英文逗号：“,”，各个字段含义如下：
article_id      法律条文对应编号    str
article_detail	法律条文具体内容	str

## **模块简介**
整个工作目录下包含三大模块：
1. 数据预处理 [data_utils]    包括[data_preparation.py](/data_utils/data_preparation.py)  [data_processing.py](/data_utils/data_processing.py)   [evaluation.py](/data_utils/evaluation.py)
2. 模型训练 [model]           [train_textcnn_keras.py]
3. 对结果的最终预测 [precditor]   "predictor.py"









**1.1数据预处理基本流程：**
1. 分别对train、test和valid数据进行分词并清洗；
2. 输入分好词的结果，使用keras的数据预处理工具把词语列表转化为词典，取频率最高的前40000个词语（可自定义）。事实证明，词典大小对结果影响较大。
3. 根据词典，利用`texts_to_sequences`功能把词语列表转为序列（数字）列表，不在词典中的词语去掉；
4. 序列的长度固定为999（也可自定义，对后续的结果也是有一定的影响），利用`pad_sequences`对序列进行截断（长度大于999）或补全（长度少于999的补0）。

**1.2数据预处理要点**：
1. 对文本进行分词进行分词，只保留长度大于1的词语（即去除单个字的）；
2. 部分案情陈述中都有的涉案金额，但金额数量比较零散，不同意，容易导致在分词后建立词典时被筛掉，所以需要对涉案金额进行化整处理；
即预先订好固定的金额区间，如“50, 100, 200, 500, 800, 1000, 2000, 5000”，然后把处于对应区间的金额转化为固定的金额数值；
3. 去掉部分停用词，查阅了部分案情陈述后发现，大部分的案情陈述都涉及相同或类似的词语，如“某某，某某乡，某某县，被告人，上午，下午”等。
这类词语词频相当高，需要把他们去掉，以免影响对数据进行干扰。

**1.3数据预处理文件**：

数据预处理的功能主要包含在`data_utils`中，包括数据预处理各类功能函数集合

[data_processing.py](/data_utils/data_processing.py)包括：
按照列分类；
去掉每个事实中都会出现的重复或类似的词语或表述；
词语分割；
文本转数字（one-hot独热码）



对数据进行实际预处理的数据准备模块[data_preparation.py](/data_utils/data_preparation.py)。
调用[data_processing.py](/data_utils/data_processing.py)
对训练集和测试集进行分词和数据化表示，存到npy格式的矩阵中。




## **2.模型**

采用基于TextCNN的NLP深度神经网络
采用  3 4 5 6 四个词窗的网络结构

## **3.进行结果预测**
用训练好的权重，在测试集上运行


## **final结果**
最终理想并不是很理想 在valid集上  micro准确率可达0.95，但macro准确率仅有0.16。后来我们分析在text2num那一步把文本转换成词向量时，我们采用的是简单的one-hot编码方式，这种转换并不能提取到文本的全部信息，所以准确率较低。









