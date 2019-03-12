
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle


# In[98]:


def load_obj(name ):
    with open('final_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def load_obj(name ):
    with open('final_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[99]:


def stock_index():
    portfolio = pd.read_excel('index_data.xlsx')
    portfolio.rename(columns = {'交易日期':'date','收盘':'price'}, inplace = True)
    portfolio = portfolio.loc[(portfolio.date>='2016-08-31')&(portfolio.date<='2018-12-01')]
    portfolio = portfolio.set_index('date', drop = True)
    return portfolio


# In[100]:


##获取数据，并作出一定的清理
def emotion_index(w1,w2):
    ##载入数据
    emotion = load_obj('emotion_data')
    emotion = emotion.reset_index()
    emotion.rename(columns = {'index':'date'}, inplace = True)
    p_change = load_obj('p_change_data')
    p_change = p_change.reset_index()
    p_change.rename(columns = {'index':'date'}, inplace = True)
    ##合并
    index_em = pd.merge(emotion,p_change, on = 'date', how = 'left')
    index_em = index_em.ffill()
    ##根据权重构建指标
    index_em['Em_index'] = w1*index_em.label+w2*index_em.Col_sum
    index_em = index_em.loc[:,['date','Em_index']]
    date = pd.to_datetime(index_em.date)
    index_em = pd.DataFrame({'Em_index':list(index_em.Em_index)}, index = date)
    return index_em


# In[101]:


#指数移动平均
def ex_average(span):
    cri = index_em.ewm(span = span).mean()
    return cri


# In[102]:


###布林线策略，用于情绪指标上
def bulin():
    df_all['mid']=df_all['Em_index'].rolling(26).mean()#为算通道做准备
    df_all['tmp2']=df_all['Em_index'].rolling(20).std()#为算通道做准备
    ##上通道
    df_all['top']=df_all['mid']+1.5*df_all['tmp2']
    ##下通道
    df_all['bottom']=df_all['mid']-1.5*df_all['tmp2']
    return df_all


# In[103]:


#突破上通道，情绪走高,由于情绪具有滞后性，买入。突破下通道，情绪走低，由于情绪具有滞后性，卖出。
def trade():
    buy = 0
    sell = 0
    sell_record = pd.DataFrame()
    buy_record = pd.DataFrame()
    for i in range(len(df_all)):
        #向上突破记为一次买入
        if df_all.top[i]<df_all.Em_index[i]:
            buy+=1
            buy_record = buy_record.append(df_all.iloc[i])
        #向下突破记为一次卖出
        if df_all.bottom[i]>df_all.Em_index[i]:
            sell+=1
            sell_record = sell_record.append(df_all.iloc[i])
    return buy_record,sell_record


# In[120]:


index_em = emotion_index(0.9,0.1)
plt.figure()
index_em.plot()
plt.title('Raw Emotion Index')
cri = ex_average(30)
plt.figure()
cri.plot()
plt.title('Emotion Index')
portfolio = stock_index()
df_all = cri.join(portfolio)
df_all.price = df_all.price.ffill()
df_all.plot(secondary_y = ['price'])




# In[233]:


#从上面的图中我发现从2018.8月开始，市场的情绪指数明显地背离股票的价格走势，我
#试图探寻其中的的原因。
#part1
#我开始在中证网上搜集新闻，发现2018.7中证网关于人工智能的新闻居然远高于平常
#达到了1000多篇，于是我从上面抓下标题，看一下关键词
#part2
#我去百度指数看与人工智能相关的搜索数据，也发现2018.7/8月搜索量显著增加。鉴于百度
#指数的爬虫难度比较大，并且没有特别大的意义，这里就没有进行了。


# In[123]:


df_all = bulin()
df_all.loc[:,['Em_index','top','bottom']].plot()
plt.title('Bulin')

# In[95]:


buy_record,sell_record = trade()
buy_record = buy_record.loc[:,'price']
sell_record = sell_record.loc[:,'price']
plt.figure()
plt.scatter(buy_record.index,buy_record)
plt.scatter(sell_record.index,sell_record)
plt.xticks(rotation = 30)
plt.legend(['buy','sell'])
plt.title('Trading')


# In[127]:


#为了让指数更平滑，我们根据价格的变动幅度，对指数的变动幅度做出限制。
#在之后的实验中，发现价格的波动率远小于指数的波动率，故对价格的波动率取了一定的倍数，5000倍
def modify(size):
    cr_index = [(df_all.Em_index.iloc[i]-df_all.Em_index.iloc[i-1])/df_all.Em_index.iloc[i-1] for i in range(1,len(df_all))]
    cr_index.insert(0,0)

    cr_price = [(df_all.price.iloc[i]-df_all.price.iloc[i-1])/df_all.price.iloc[i-1] for i in range(1,len(df_all))]
    cr_price.insert(0,0)
    
    rate = []
    num = 5
    for i in range(num,len(cr_index)):
        rate_limit = size*sum([abs(cr_price[j]) for j in range(i-num,i)])/num
        if abs(rate_limit)<= 0.1 :
            rate.append(cr_index[i])
        else:
            if abs(cr_index[i])<=rate_limit:
                rate.append(cr_index[i])
            else:
                if cr_index[i] >=0:
                    rate.append(rate_limit)
                else:
                    rate.append(-rate_limit)
    remain = cr_index[0:num]
    remain.append(rate[0])
    rate[0:1] = remain
    count = 0
    for i in range(len(rate)):
        if rate[i] != cr_index[i]:
            count +=1

    #根据调整后的指数变动率，算出新的每日指数
    init = df_all.Em_index.iloc[0]
    Em_index_modified = [init]
    for i in range(1,len(rate)):
        init = init*(1+rate[i])
        Em_index_modified.append(init)
    return Em_index_modified,count


# In[143]:


Em_index_modified,count = modify(6000)
df_all['Em_index_modified'] = Em_index_modified
df_all.loc[:,['Em_index_modified','price']].iloc[100:].plot(secondary_y = ['price'])
#可以显著地发现情绪指标非常不稳健

