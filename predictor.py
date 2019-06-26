# -*- coding: utf-8 -*-
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
 
from data_utils.data_processing import data_process



localpath = os.path.dirname(__file__)

class Predictor(object):
    def __init__(self, num_words=40000, max_len=999,
                 path_tokenizer='./variables/tokenizer_40000.pkl', #token地址
                 path_accusation='./model/textcnn_accusation_token_40000_pad_999_filter_512_hidden_1000_epoch_1_accu_95_f1_55.h5',  #accusation权重位置
                 path_relative_articles='./model/textcnn_articles_token_40000_pad_999_filter_512_hidden_1000_epoch_1_accu_92_f1_55.h5'): #articles权重位置
        self.num_words = num_words
        self.max_len = max_len
        self.path_accusation = path_accusation
        self.path_relative_articles = path_relative_articles
        self.content_process = data_process()
        self.path_tokenizer = path_tokenizer
        self.model_relative_articles = load_model(path_relative_articles)
        self.model_accusation = load_model(path_accusation)

    def predict(self, content):
        content_process = self.content_process
        content_seg = content_process.segmentation(content, cut=True, word_len=2, replace_money_value=False,
                                                   stopword=True)

        with open(self.path_tokenizer, mode='rb') as f:
            tokenizer = pickle.load(f)

        content_process.text2num(content_seg, tokenizer=tokenizer)
        content_seg_num_sequence = content_process.num_sequence
        content_fact_pad_seq = pad_sequences(content_seg_num_sequence, maxlen=self.max_len, padding='post')
        content_fact_pad_seq = np.array(content_fact_pad_seq)

        model_relative_articles = self.model_relative_articles
        relative_articles = model_relative_articles.predict(content_fact_pad_seq)
        model_accusation=self.model_accusation
        accusation=model_accusation.predict(content_fact_pad_seq)
        
        def predict2tag(predictions, sets):
            m = []
            for x in predictions:
                x_return = sets[x > 0.5]
                if len(x_return) == 0:
                    x_return = sets[x == x.max()]
                m.append(x_return)
            return m

        acc_set_labels = np.load('./variables/label_set/set_accusation.npy')
        art_set_labels = np.load('./variables/label_set/set_articles.npy')
        acc_result = predict2tag(accusation, acc_set_labels)
        art_result = predict2tag(relative_articles, art_set_labels)

        r = pd.DataFrame({'ids': ids, 'accusation': acc_result, 'articles':art_result })
        r.to_csv('./results/final_result.csv',index=0,encoding='utf-8')

#执行结果输出
output=Predictor()
inf=pd.read_csv("./predict/data/test.csv",header=0)
fact=inf['fact']
ids=inf['ids']
output.predict(fact)
