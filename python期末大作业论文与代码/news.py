
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import jieba
import collections


# In[2]:


#这里并不打算编写一个通用的爬虫，所以很多地方直接特殊化了，如最大页码
def get_title():
    ls_title = []
    for page in range(1,105):
        url = 'http://search.cs.com.cn/search?page='+str(page)+'&channelid=215308&searchword=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD&keyword=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD&token=12.1462412070719.47&was_custom_expr=DOCTITLE%3D%28%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%29&perpage=10&outlinepage=5&&andsen=&total=&orsen=&exclude=&searchscope=DOCTITLE&timescope=&timescopecolumn=&orderby=&timeline==2018.07'
        html = requests.get(url)
        bsobj = BeautifulSoup(html.text, 'lxml')
        bsobj = bsobj.tr.find('td',{'class':'searchresult'}).findAll('table')
        ls_bsobj = []
        for i in range(len(bsobj)):
            if i%3 == 0:
                ls_bsobj.append(bsobj[i])
        ls_bsobj = ls_bsobj[0:10]
        for i in ls_bsobj:
            ls_title.append(i.a.text)
        if page%10 == 0:
            print(page)
    return ls_title


# In[3]:


def keywords(ls_title):
    #初步分词
    ls_jieba_title = []
    jieba.add_word('科大讯飞')
    for i in ls_title:
        ls_jieba_title.extend(list(jieba.cut(i)))
    #设置停用词表
    with open('chinese_stop_words.txt',encoding = 'utf-8') as f:
        f = f.read()
    f = f.split('\n')
    f[0] = '，'
    f.append(' ')
    f.append(',')
    f.append('【')
    f.append('】')
    #获取关键词
    ls_filtered = []
    for i in ls_jieba_title:
        if i not in f:
            ls_filtered.append(i)
    dic = collections.Counter(ls_filtered)
    return sorted(dic.items(), key = lambda x:x[1], reverse = True)


# In[4]:


ls_title = get_title()
key_word = keywords(ls_title)
print(key_word)


# In[5]:


key_word


# In[75]:


#从keyword中我们可以看到新闻报道的标题几乎都为积极性的标题，希望带动市场情绪，但是我们看到
#实际上市场情绪在不断地波动，这也是受到了诸如科大讯飞这类人工智能龙头股的丑闻曝光的影响。
#股价一路下跌，可能是出于避险的原因，大股东大量出货，散户观望或者跟风出货。
#但是最近受到政策因素的影响股价又开始往上涨了。

