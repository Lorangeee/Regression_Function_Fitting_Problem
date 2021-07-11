#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
import model

x = np.random.randn(1,100)
y = np.power(x,2)+x+1

costs,W1 = model.model(x,y,1000,0.005)

with open('test_data.txt', 'w') as f:
    f.write('{')#这样子字典没有自动的大括号要自己加
    for key in costs:
        f.write('\n')
        f.writelines('"' + str(key) + '": ' + str(costs[key]))
    f.write('\n'+'}')
    
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功") 

text_save('/Users/lorange/Desktop/test_data.txt',W1)


# In[ ]:




