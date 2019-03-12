
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import os
from gensim.models import word2vec
import logging
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM,Masking,GRU,Bidirectional,Dropout
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)#这是一条用word2vec的必须命令，什么用我也不是很知道


# In[2]:


def save_obj(obj, name ):
    with open('word2vec/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[3]:


df_train_data = load_obj('df_train_data')
df_predict_data = load_obj('df_predict_data')


# ## 训练word2vec

# In[5]:


def train_word2vec():
    ##合并两组数据，让word2vec模型的数据量更大更全
    df_all_data = pd.concat([df_train_data.content,df_predict_data.content], ignore_index = True).to_frame()
    ls_all_data = list(df_all_data.content)#转换为列表形式方便训练
    ##训练并保存word2vec
    model = word2vec.Word2Vec(ls_all_data,size = 200, min_count = 5,workers = 20)#20线程
    model.save('./word2vec/word2vec.model')


# In[5]:


train_word2vec()


# ## 向量化

# In[6]:


#用来把分完词的句子向量化的函数
def vec_lize(ls):
    newls = []
    for words in ls:
        templs = []
        for word in words:
            try:
                templs.append(model[word])
            except:
                pass
        newls.append(templs)
    return newls 


# #### 标签样本向量化

# In[7]:


model = word2vec.Word2Vec.load('./word2vec/word2vec.model')
##随机打乱，为后续分训练，测试，验证集做准备
df_shuffled = df_train_data.sample(frac = 1)
##分出训练集，测试集，验证集
a = len(df_train_data)
xunlian = list(df_shuffled.iloc[0:int(a*0.8)].content)
xunlian_label = list(df_shuffled.iloc[0:int(a*0.8)].label)
ceshi = list(df_shuffled.iloc[int(a*0.8):int(a*0.9)].content)
ceshi_label = list(df_shuffled.iloc[int(a*0.8):int(a*0.9)].label)
yanzheng = list(df_shuffled.iloc[int(a*0.9):a].content)
yanzheng_label = list(df_shuffled.iloc[int(a*0.9):a].label)
##向量化
xunlian = vec_lize(xunlian)
ceshi = vec_lize(ceshi)
yanzheng = vec_lize(yanzheng)


# ## GRU训练

# In[8]:


def deeplearning():
    #参数设定
    BATCH_SIZE = 100   #每次训练多少句话
    TIME_STEPS = 30   #一句话多少个词向量
    INPUT_SIZE = 200   #每个词向量的长度
    OUTPUT_SIZE = 4  #label的宽度
    LR = 0.001

    #样本标签one hot 化
    y_train = np_utils.to_categorical(xunlian_label, num_classes = OUTPUT_SIZE)
    y_test = np_utils.to_categorical(ceshi_label, num_classes = OUTPUT_SIZE)
    y_validation =  np_utils.to_categorical(yanzheng_label, num_classes = OUTPUT_SIZE)

    #统一词向量长度
    x_train = pad_sequences(xunlian, maxlen = TIME_STEPS, padding ='post', dtype = 'float')
    x_test = pad_sequences(ceshi, maxlen = TIME_STEPS, padding ='post', dtype = 'float')
    x_validation = pad_sequences(yanzheng, maxlen = TIME_STEPS, padding ='post', dtype = 'float')

    #模型构建
    #我试了RNN，LSTM，GRU发现GRU的效果相对比较好
    model = Sequential()
    model.add(Masking(mask_value = 0,input_shape = (TIME_STEPS,INPUT_SIZE)))#这里mask_value参数去除了输入的词向量中的零向量实现了GRU的变长度输入
    model.add(Bidirectional(GRU(64)))#这里可选的参数常见有32,64,128,256我试了以后发现32比较好
    model.add(Dropout(0.5))#这是对GRU门的设置，为了防止过拟合
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    adam = Adam(LR)
    model.compile(optimizer = adam, 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])

    #跑模型
    result = model.fit(x_train,y_train, batch_size=BATCH_SIZE,
                       nb_epoch=5, verbose=1, validation_data=(x_test, y_test))
    
    #评估
    score, acc = model.evaluate(x_validation, y_validation, batch_size = BATCH_SIZE, verbose = 1)
    print('五分类下的准确率{}'.format(acc))

    # 鉴于上述四分的时候准确率不高，我们来看一下二分的时候
    validation_label = model.predict_classes(x_validation,batch_size = BATCH_SIZE)
    num = 0
    for i in range(len(x_validation)):
        if validation_label[i] <= 1 and yanzheng_label[i] <= 1:
            num +=1
        if validation_label[i] >=2 and yanzheng_label[i]>=2:
            num+=1
    print('二分法下的准确率{}'.format(num/len(x_validation)))#四分类不是很准，但是对市场情绪的积极，消极判断还是不错的

    model.save('./GRU/gru.h5')


# ## 获得预测样本情绪

# In[15]:


def index_construction():
    #导入模型，获得预测样本的情感标签
    model = load_model('./GRU/gru.h5')
    TIME_STEPS = 30
    x_predict = pad_sequences(predict, maxlen = TIME_STEPS, padding ='post', dtype = 'float')
    predict_label = model.predict_classes(x_predict)
    df_predict_data['label'] = predict_label
    #因为在做情感标记的时候为了onehot化方便，把情感从0开始标记。
    #这里我们要更改标记为-2,-1,0,1,2
    df_predict_data.label = df_predict_data.label.apply(lambda x: x-2 if x<=1 else x-1)
    #获得每日的情感分数
    df_final = df_predict_data.groupby('date').label.sum().to_frame()
    #每日情感分数除每日收集的评论总数
    df_count = df_predict_data.groupby('date').label.count()
    df_final.label = df_final.label/df_count
    #标准化
    av = df_final.label.mean()
    std = math.sqrt(df_final.label.std())
    df_final.label = df_final.label.apply(lambda x: (x-av)/std)
    df_final = df_final.loc['2016-08-31':'2018-12-01']
    with open('final_data/'+ 'emotion_data' + '.pkl', 'wb') as f:
        pickle.dump(df_final, f, pickle.HIGHEST_PROTOCOL)


# In[10]:


deeplearning()


# #### 预测样本向量化

# In[11]:


#我发现如果这段写成函数，model老是莫名其妙报错，所以就没有写成函数了
model = word2vec.Word2Vec.load('./word2vec/word2vec.model')
predict = list(df_predict_data.content)
predict = vec_lize(predict)#向量化后的文本


# In[ ]:


index_construction()

