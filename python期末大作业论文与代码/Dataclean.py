
# coding: utf-8

# In[3]:


import re
import pandas as pd
import numpy as np
import os
import jieba
import pickle
import math
import string
import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 基本函数

# In[4]:


#用来存列表/字典的
def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#用来读列表/字典的
def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# 得到所选股票的股票代码存到 stock_name_list中
def get_stock_name_list():
    name = pd.read_table('stock_list.txt')
    name['品种代码'] = name['品种代码'].apply(lambda x:x[0:-3])
    stock_name_list = [i for i in name['品种代码']]
    return stock_name_list
stock_name_list = get_stock_name_list()

#在分词以后，针对每一句话（列表形式）去除其中的数字，标点等
filter_word = string.punctuation+string.ascii_letters+string.digits+'！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
def clean_predict_punc(ls):
    new_ls = []
    for i in ls:
        if i not in filter_word:
            try:
                float(i)
            except:
                new_ls.append(i)
    return new_ls

#从之前存储评论的文件夹中获取评论
def get_data(types):
    df_all = pd.DataFrame(columns = ['date','content'])#一个空的DataFrame
    for stock_code in stock_name_list:
        newstore_predict_address = stock_code+'/'+stock_code+ types#构造地址,如000055/000055predict.pkl,如果types = predict
        df = load_obj(newstore_predict_address)
        df_all = df_all.append(df, ignore_index = True)#把这支股票的评论加到df_all里面，最后让df_all包含所有股票的带预测的评论
    return df_all

#分词函数(包括分词后的清理)
def jieba_and_clean(df):
    ##分词
    df.content = df.content.apply(lambda x: list(jieba.cut(x)))#jieba分词并返回列表形式
    ##清理
    df.content = df.content.apply(lambda x: clean_predict_punc(x))#去除列表中的标点，数字等
    df.content = df.content.astype(str)
    df = df.loc[df.content != '[]']#去除空列表
    df.content = df.content.apply(lambda x: eval(x))
    return df


# ## 预测数据清理

# In[5]:


def predict_data_dispose():
    ##获取全部预测评论
    df_predict = get_data('predict')
    ##数据清理
    df_predict = df_predict.sort_values(['date'], ascending = True)#将评论按时间排序
    df_predict = df_predict[df_predict.date != '00']#去除缺失的评论（在爬虫中我把因为没抓下来的评论赋值成00）
    df_predict = df_predict[df_predict.date >= '2016-08-31']#去除日期小于2016-08-31的评论
    df_predict = df_predict.reindex(range(len(df_predict)))#因为评论按照时间排序了，index乱了，为了好看点重排index
    ##分词和进一步清理
    df_predict.content = df_predict.content.astype(str)#为了jieba分词做准备
    df_predict = jieba_and_clean(df_predict)
    ##储存
    save_obj(df_predict, 'df_predict_data')#将处理完成的列表数据保存起来
    #此处改为
    #return df_predict
    #就可以看到处理的结果了


# ## 标签数据
# - expression_dict 得到每个表情对应的话
# - word_dict 得到每句话中包含的表情

# In[8]:


#获取表情符号
def gain_expression_list():
    df_train = get_data('train')
    del df_train['date']
    expression = {}
    rdgx = re.compile('(\[[^\[\]]*\])')
    for i in range(len(df_train)):
        comment = df_train.iloc[i,0]
        expr_ls = rdgx.findall(comment)
        expr_ls = [x[1:-1] for x in expr_ls]
        for i in expr_ls:
            if i in expression:
                expression[i] +=1
            else:
                expression[i] = 1
    expression = sorted(expression.items(), key = lambda x:x[1], reverse = True)
    expression = expression[0:41]
    expression_list = []
    for i in expression:
        expression_list.append(i[0])
    expression_list.remove('图片')
    return expression_list,expression

def train_data_dispose():
    ##获取全部标签评论
    df_train = get_data('train')
    del df_train['date']#因为训练样本不需要日期，故删除
    ##得到每句话包含的表情
    for i in expression_list:
        expression_dict[i] = []
    word_dict = {}#用来存每句评论包含的表情符号，key = 评论，item = 包含表情（列表形式）
    rdgx = re.compile('(\[[^\[\]]*\])')#用来匹配表情符号，如 [买入]
    for i in range(len(df_train)):
        comment = df_train.iloc[i,0]
        expr_ls = rdgx.findall(comment)
        expr_ls = [x[1:-1] for x in expr_ls]
        comment= rdgx.sub('',comment)#去除comment中的表情符号
        if comment !='':
            if comment in word_dict:
                pass#如果这句评论有了，就不加入了
            else:
                word_dict[comment] = expr_ls #向word_dict里面加东西，key = 评论，item = 包含表情（列表形式）
            for i in set(expr_ls):
                if i in expression_dict:
                    expression_dict[i].append(comment)#向expression_dict里面加东西
    ##储存
    save_obj(expression_dict, 'expression_dict')
    save_obj(word_dict,'word_dict')
    ##返回
    return expression_dict,word_dict


# In[9]:


expression_list,expression = gain_expression_list()
expression_dict = {}#用来存每个表情对应的评论
predict_data_dispose()#这个函数跑得很慢
expression_dict,word_dict = train_data_dispose()


# ## 确定每个表情的情感分数

# In[10]:


##随机抽样（总共40个表情符号，每个表情符号各抽取50句评论人工打分）
#此部分属于纯体力活，时间紧迫，抽取的评论比较少还请老师见谅，呜呜
def sample_choose(expression_dict): 
    sample_dict = {}
    for i in expression_dict:
        sample_dict[i] = random.sample(expression_dict[i],50)
    return sample_dict
#这里举个例子，给带不说了这个表情打分
'''
sample = sample_choose(expression_dict)
a = sample['不说了']
for i in range(50):
    print(a[i])
'''

# ### 打分

# In[11]:


#这里是根据一次抽样，我手工打上的标签，p积极，0中立，n消极

a1 = '大笑   nnnppn0nppnnnpn0npnnpnn0000p0np00pn0pp00p00ppnn0pp'
a2 = '献花   pppppppppppppppppppppnp00ppppppnp0pnpppppppppnpn00'
a3 = '胜利   n0npnpppppnpp0000n00npppppnppppnn00pp000000nnppp00'
a4 = '鼓掌   pn0nn000000000n0p00pp000p0p00ppppnppnppp0pnppp00n0'
a5 = '微笑   pnnpnnnn000000ppppp0nppp000000np000000n000npnnnnn0'
a6 = '哭     nnnnpppnnnnnnn0nn0nnnnn0nnpn0n0n0nnnnpnn0nnnpppnnn'
a7 = '赞     00pn0npp0p0pp00n0pn00p000n0ppppn0n0pp0pnpnpp0npn00'
a8 = '不赞   p00nn0nn00nnnn0nnn0nnnnnnnnnnn00nnn00nnnnnnnnnnn0n'
a9 = '大便   0nnnn0n0nnnnnnnnnnnpnnnnnpnnnnn0nnnnnnnnn0nnnnn0pn'
a10 = '买入  ppppppppppppppppp0pppppppp0ppppppp00ppppppnppnppnp'
a11 = '不屑  00np0nnnnnnpn0nnnn0nnn00pnnnnn0pnnn0nppnnnnn00nnpn'
a12 = '滴汗  np0n0nppp0000npnnnnnn0n0nnnnnn0nn0npnppnnnnnp000pp'
a13 = '抄底  0n00pppp0np000ppp00p0ppppppp0000ppp0p0pp0ppnnp0pnp'
a14 = '牛    0pppppppppppppppppp00p0ppppp0pppppp000pppp0p0ppppp'
a15 = '摊手  ppppnpnnn0pnnn0n0npn0n00nnnnpnp0nnnnn0nnnnn0n00000'
a16 = '拜神  p0p0p000pn000pppppp00nnppnn0nn0npppnppp00n00p0p0p0'
a17 = '加油  p000pp0n0ppppn0pnnp0pp0p0n000np0ppnnnn000pp000nnnp'
a18 = '加仓  ppp000pppppppppppppp0p00ppppppppp0p0ppppppppppppp0'
a19 = '傲    00000nnnn0n0npn0nnp0p00n00nn000npnn0nnn0nnpnnnn0n0'
a20 = '俏皮  npppp000000nnp0np0n0pppppp00000000pn0pnppn0ppppnpp'
a21 = '卖出  nnnnnnnnnnnnnnnnnn0nn0nnnnnnnnnnnnnnnn0nnnnnnnnnnn'
a22 = '怒    nnnnnnnnnnnnnnnnnnnnnnnnnnnnn0n0n0nnnnnnnnnnnnnnnn'
a23 = '满仓  ppppppn0pppppppppppppppppppp0p0p0nppnp00nppppp0ppp'
a24 = '兴奋  pp0p00nnpppnnn0nnnppnp00npnp0n0nnnppnnnnnp0pn0pppp'
a25 = '心碎  0npnnnnnnnnnnnnnnnnnpnnnnnnnnnnnnnnnnnnnnnnnnnnnnn'
a26 = '成交  np0ppp0n0nn0ppnnnnn00n00pn000nnnnnpnnnpn0000npn0np'
a27 = '失望  nnnnnnnn00nnnnnnnnn00nnnn0nnnnnnnnnnn000nnnnnnnnnn'
a28 = '围观  0000n00nnpnpp0nn000n0000000nnp0n0p00n000np0npnp0n0'
a29 = '空仓  nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn'
a30 = '困顿  0n0000nn0nn000000n000n000nnn0n0n0000n0n00000nn0nnn'
a31 = '财力  ppppppppppp0ppppppppp0nnppp0ppppppppppn00pppnnpppp'
a32 = '好逊  0n00nn0nn0n00nn0nn00n0n0npp000nnpnn0n0nnnnnnnnnnn0'
a33 = '不说了00000nnnp00000000nnp0pn000n0n000ppnnn000p000000000'
a34 = '看空  nnnnnnn0n0nnnnnnnn00n0nn0nnnn00nnnnnnnnnnnnnnn0000'
a35 = '想一下nnp00npnn0n000n0nnnn0pnnn00nnp0n00n00ppnn00000p000'
a36 = '亏大了np0pnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn'
a37 = '为什么n000n00pp00000p00000000pn00nppp0n0pn000000n0pn00pp'
a38 = '好困惑nnnp0nnnpnn0nnn0000000000n0n000000nn00nnn0nnn00npp'
a39 = '赚大了pppppppppppppppppppppppppppppppppppppppppppppppppp'
a40 = '看多  pppppppppppppppppppppppppppppppppppppppppppppppppp'


# In[12]:


#对上面打的标签的结果进行一定的处理
def expression_stat(ex):
    expression = ex[0:3].rstrip(' ')#获取表情名称
    expression_stats = ex[-1:-51:-1]
    count_n = 0#数积极情绪的标签数
    count_zero = 0#数中立情绪的标签数
    count_p = 0#数消极情绪的标签数
    trust = []
    stat_ls = []
    for i in expression_stats:
        if i =='n':
            count_n +=1
        if i =='0':
            count_zero +=1
        if i =='p':
            count_p +=1
    #在这里我根据这个表情中积极情绪和消极情绪出现的次数作比较，将表情的极性分为积极和消极，
    #分值按照： 积极情绪出现的次数/50   消极情绪出现的次数/50
    if count_p>= count_n:
        trust = [1,count_p/50]
    else:
        trust = [-1, count_n/50]
    stat_ls = [count_n, count_zero, count_p]
    stat_ls.extend(trust)
    stat_ls.insert(0,expression)
    return stat_ls

def expression_mark():
    ls_all = []
    createvar = globals()#获取此时的全局变量（字典形式），这是一个非常重要的技巧
    for i in range(1,41):
        ls_all.append(expression_stat(createvar['a'+str(i)]))#从全局变量字典中取出变量
    fm = '{}\t{}\t{}\t{}\t{}\t{}'
    print(fm.format('表情','积极个数','中立个数','消极个数','极性','分值'))
    for i in ls_all:
        print(fm.format(i[0],i[1],i[2],i[3],i[4],i[5]))
    return ls_all


# In[13]:


ls_all = expression_mark()


# ## 得到每句话的情感分数

# In[14]:


def gain_df_label():
    ##得到一个表情词典，key=表情符号 value=分值
    expression_label_dict = {}
    for i in ls_all:
        expression_label_dict[i[0]] = i[-2]*i[-1]
    ##遍历word_dict，得到其中每句话的情感分值，存入word_label_dict
    word_label_dict = {}
    for i in word_dict:
        ex_ls = word_dict[i]
        ex_ls = [x for x in ex_ls if x in expression_list]
        label = sum([expression_label_dict[y] for y in ex_ls ])
        word_label_dict[i] = label
    ##转化为DataFrame
    label = list(word_label_dict.values())
    words = list(word_label_dict.keys())
    df_label = pd.DataFrame()
    df_label['content'] = words
    df_label['label'] = label
    return df_label


# In[15]:


df_label = gain_df_label()


# ### 训练集构建

# In[122]:


#df_label.label.hist(bins = 100)画一下分值的直方图来直观感受一下


# In[16]:


def pct(num):
    return np.percentile(df_label.label,num) 

#分类函数
def label_classify(df_label):
    pct25 = pct(20)
    pct50 = pct(40)
    pct75 = pct(60)
    for i in range(len(df_label.label)):
        label = df_label.iloc[i,1]
        if label <= pct25:
            df_label.iloc[i,1] = 0
        elif pct25<label<pct50:
            df_label.iloc[i,1] = 1
        elif pct50<=label<pct75:
            df_label.iloc[i,1] = 2
        elif label>+pct75:
            df_label.iloc[i,1] = 3
    ##分词，去标点,去数字
    df_label = jieba_and_clean(df_label)
    ##储存
    save_obj(df_label,'df_train_data')
    #此处改为
    #return df_label
    #就可以看到处理的结果了


# In[17]:


label_classify(df_label)

