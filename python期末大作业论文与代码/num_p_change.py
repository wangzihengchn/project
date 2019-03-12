
# coding: utf-8

# In[1]:


import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import pickle


# In[2]:


def get_stock_name_list():
    name = pd.read_table('stock_list.txt')
    name['品种代码'] = name['品种代码'].apply(lambda x:x[0:-3])
    stock_name_list = [i for i in name['品种代码']]
    return stock_name_list


# In[9]:


#获取p_change的函数
def get_p_change():
    ls = []
    ##从tushare上获取股票池每日的涨跌幅，这里我把涨跌幅超过7%的股票分别记为1/-1，求和
    for i in range(len(stock_name_list)):
        code = stock_name_list[i]
        a = ts.get_hist_data(code, start = '2016-08-31', end = '2018-12-01').loc[:,'p_change'].to_frame()
        a.rename(columns = {'p_change':code}, inplace = True)
        a.loc[(a[code]<7)&(a[code]>-7)] = 0
        a.loc[a[code]>=7] = 1
        a.loc[a[code]<=-7] = -1
        ls.append(a)
    df_p_change = pd.concat(ls,axis=1)
    df_p_change['Col_sum'] = df_p_change.apply(lambda x: x.sum(), axis=1)#对行求和，得出一日内股票池的涨跌停板加和
    df_p_change = df_p_change['Col_sum'].to_frame()
    return df_p_change


# In[5]:


#标准化处理函数
def dispose_p_change():
    max_90 = df_p_change.Col_sum.sort_values().iloc[int(0.9*len(df_p_change))]#找到90%分位的数
    min_10 = df_p_change.Col_sum.sort_values().iloc[int(0.1*len(df_p_change))]#找到10%分位的数
    ## 去除离群值函数
    def change(x):
        if x> max_90:
            return max_90
        if x< min_10:
            return min_10
        else:
            return x
    df_p_change.Col_sum = df_p_change.Col_sum.apply(lambda x: change(x))
    ##z-score标准化方法
    av = df_p_change.Col_sum.mean()#平均数
    std = math.sqrt(df_p_change.std())#标准差
    df_p_change.Col_sum = df_p_change.Col_sum.apply(lambda x: (x-av)/std)
    return df_p_change


# In[6]:


def save_obj(obj, name ):
    with open('final_data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[10]:


stock_name_list = get_stock_name_list()
df_p_change = get_p_change()
plt.figure()
plt.plot(df_p_change)
plt.xlabel('date')
plt.ylabel('raw_p_change')


# In[8]:


df_p_change = dispose_p_change()
plt.figure()
df_p_change.Col_sum.plot()
plt.xlabel('date')
plt.ylabel('disposed_p_change')


# In[64]:


save_obj(df_p_change,'p_change_data')

