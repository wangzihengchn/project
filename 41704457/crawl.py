
# coding: utf-8

# In[17]:


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import os
import pickle


# In[18]:


def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[19]:


# 得到所选股票的股票代码存到 stock_name_list中
def get_stock_name_list():
    name = pd.read_table('stock_list.txt')
    name['品种代码'] = name['品种代码'].apply(lambda x:x[0:-3])
    stock_name_list = [i for i in name['品种代码']]
    return stock_name_list



# ## 获取网页

# In[20]:


#用来解析一页股吧网页的函数，获取评论和日期
#这里比较麻烦的是，解析一个页面的时候我们只能获取评论发表的月与日，无法获取年。
#这里又两者解决办法：一是可以再进入这条评论的url，但是这样子每个评论都要重新加载网页，非常慢
#                    二是可以利用月份循环的特点进行判断，我们采用这种方法。
def get_one_page(url,yr):   
    html = requests.get(url)
    bsobj = BeautifulSoup(html.text,'lxml')
    bsobj = bsobj.find('div',{'id':'articlelistnew'}).findAll('div',{'class': re.compile(r'^articleh normal_post')})
    ls = []
    for i in bsobj:
        try:
            content = i.find('span',{'class':'l3'}).a.attrs['title']#评论
            #进行了年份的判断
            date = yr+'-'+i.find('span',{'class':'l6'}).text#日期，加上了年份
        except:
            #虽然股吧的评论页面整体非常规整，但是仍有可能有报错的情况，这里对报错情况进行了处理
            content = '0'
            date = '00'
        ls.append([date,content])
    return date, ls


# In[21]:


#获取整个股票的从2016/8/31到2018/12/1的评论，难点在于判断年份
def get_stock(stock_code):
    page = 0
    temp = 0
    year = ['2018','2017','2016']
    ls = []
    content_all = []
    while True:    
        page += 1
        url = 'http://guba.eastmoney.com/list,'+stock_code+',f_'+str(page)+'.html'
        try:
            date,content_one_page = get_one_page(url,year[temp])
        except:
            break
        month = date[5:7:1]
        if month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            ls.append(month)
        if temp == 2 and len(ls) == 6:
            break
        else:
            if len(ls) == 13 and month == '12':
                ls = ['12']
                temp +=1
            else:
                ls = list(set(ls))
        if page%20 == 0:
            print(page)
        content_all.extend(content_one_page)
    return content_all


# ## 得到所有股票信息

# In[403]:


def gain_and_store(stock_code):
    df = pd.DataFrame(get_stock(stock_code))
    df.rename(columns = {0:'date',1:'content'}, inplace = True)
    df['fil'] = df.content.str.extract('(?P<content>\[.*\])').notnull()
    df_train = df.loc[df.fil]
    df_predict = df.loc[df.fil == False]
    del df_train['fil']
    del df_predict['fil']
    newfile_address = './data/'+stock_code
    os.mkdir(newfile_address)
    newstore_train_address = stock_code+'/'+stock_code+'train'
    newstore_predict_address = stock_code+'/'+stock_code+'predict'
    save_obj(df_train, newstore_train_address)
    save_obj(df_predict, newstore_predict_address)


# In[406]:

stock_name_list = get_stock_name_list()
for i in stock_name_list[39:101]:
    gain_and_store(i)
    print(i)


