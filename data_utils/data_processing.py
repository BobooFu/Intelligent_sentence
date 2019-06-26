# -*- coding: UTF-8 -*-
import pandas as pd
import os
import jieba
import numpy as np
import re
import json
from keras.preprocessing.text import Tokenizer

jieba.setLogLevel('WARN')


class data_process(object):
    def __init__(self):
        self.data_label = {}
        self.label_collection = {}
        self.label_type = {'articles','accusation'}
        self.decomposition = {}
        self.shape=(0,0)


    def decompose_data(self,name='',filename='None'): 
        '''
        去csv中的列
        name为列名
        filename为读取的文件名
        '''
        f=pd.read_csv(filename,header=0)
        if name =='fact':
            decomposition=f['fact']
        elif name =='accusation':
            decomposition=f['accusation']
        elif name =='articles':
            decomposition=f['articles']
        self.decomposition.update({name:decomposition})


    def data_cleaning(self, lines):
        '''
        去掉每个事实中都会出现的重复或类似的词语或表述
        :param lines: 列表（文本列表）
        :return:
        '''
        # 去掉日期的数字
        date_list = [str(i) for i in np.array(range(0, 31)) + 1]
        year_list = [str(i) for i in np.array(range(2000, 2018)) + 1]
        stopword_list = ['某某', '某.+', r'.+省\b', r'.+市\b', r'.+某\b', r'.+市\b', r'.+区\b', '被告人',r'xx\b',
                         '有限公司', '分公司', r'.+乡\b', '人民检察院', '指控', r'.+县\b', '时许', '下午', '上午']
        stopword_pattern = '|'.join(stopword_list)
        lines1 = []
        for sentence in lines:
            sentence1 = []
            for word in sentence:
                if word in date_list or word in year_list:
                    continue
                elif re.findall(stopword_pattern, word) != []:
                    continue
                else:
                    sentence1.append(word)
            lines1.append(sentence1)
        return lines1

    def segmentation(self, list=None, cut=True, word_len=1, path=None, replace_money_value=False, stopword=False):
        '''
        对文本列表进行分词 / seperate each word of the data in the text list
        :param list: 列表（原文本）s
        :param cut: 是否需要分词
        :param word_len: 保留的词语长度
        :param path: 保存路径
        :param replace_money_value: 是否按把钱的值替换为给定值
        :param stopword: 是否去掉部分停用词
        :return:
        '''
        if cut == False:
            seg_list = [[word for word in text if len(word) >= word_len] for text in list]
        else:
            if replace_money_value == True:
                seg_list = [[word for word in jieba.lcut(self.replace_money_value(text)) if len(word) >= word_len] for
                            text in list]
            else:
                seg_list = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in list]
        if stopword == True:
            seg_list = self.data_cleaning(seg_list)
        if path != None:
            with open(path, 'w') as f:
                json.dump(seg_list, f)
        return seg_list

    def get_label_collection(self, label_type=None):
        '''
        读取标签的列表，包括法律和罪名
        :param type: 取值可为law、accu或None,如果不为None时，多个输入需加上下划线
        :return:
        '''
        if label_type == None:
            labels = self.label_type
        else:
            labels = label_type.split('_')

        for label in labels:
            with open('./good/%s.txt' % label, 'r', encoding='utf-8') as f:
                label_set = f.read().split()
            self.label_collection.update({label: label_set})

    def text2num(self, text_list=None, tokenizer=None, num_words=40000):
        '''
        :param text_list: 取出长度为2后的json
        :param tokenizer: train集该参数为空，生成token
        :param num_words: 词库的大小
        :return:
        '''
        list_length = len(text_list)
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=num_words)
            if list_length > 10000:
                print('文本过多，分批转换')
            n = 0
            # 分批训练
            while n < list_length:
                tokenizer.fit_on_texts(texts=text_list[n:n + 10000])
                n += 10000
                if n < list_length:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % list_length)
            self.tokenizer = tokenizer

        # 全部转为数字序列
        num_sequence = tokenizer.texts_to_sequences(texts=text_list)
        self.num_sequence = num_sequence
        print('finish texts to sequences')

        # 内存不够，删除
        del text_list

    def one_hot(self, label, label_set):
        '''
        :param label: 每一条数据对应要预测的标签
        :param label_set: 该标签列表对应的总标签类
        :return: 输出bool array
        '''
        one_hot = (np.in1d(np.array(label_set), label) + 0).reshape(1, -1)
        return one_hot

    def transform_label(self, labels=None, label_type='articles'):
        '''
        :param labels: 待处理列向量
        :param label_type: 待处理列名称
        :return: np数组
        '''
        label_one_hot = np.concatenate([self.one_hot(label, self.label_collection[label_type]) for label in labels])
        return label_one_hot