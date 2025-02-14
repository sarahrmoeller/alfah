#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import csv
import os
import glob


# In[132]:

path1 = 'new_texts_for_tagging'
txt_files = glob.glob(path1 + '/*.txt')
path2 = 'speakers'
speaker_files = glob.glob(path2 + '/*.txt')
for f in txt_files:  
    data = pd.read_csv(f, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', engine = 'python', header = None)
    for s in speaker_files:
        ind_f1 = f.index('/')
        ind_f2 = len(f) - f[::-1].index('_')-1
        name_f = f[ind_f1:ind_f2]
        ind_s1 = s.index('/')
        ind_s2 = s.index('_')
        name_s = s[ind_s1:ind_s2]
        if (name_f == name_s):
            speaker = pd.read_csv(s, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', engine = 'python', header = None)
            print('yes')
            print(name_f)
            print(name_s)
        else:
            print('no')
            print(name_f)
            print(name_s)
            
        
# In[133]:

    data.columns = ['Sentence']
    speaker.columns = ['Speaker']

    # In[134]:


    #data = data.insert(0, "speaker", speaker)
    data = pd.concat([speaker, data], axis = 'columns')

    # In[135]:


    speaker.loc[1, 'Speaker']


    # In[ ]:





    # In[136]:


    count = 0
    for i in range(len(data)):
        index = data.loc[i, 'Sentence'].find(':')
        if index >= 0:
            count += 1
        data.loc[i, 'Speaker'] = speaker.loc[count, 'Speaker']

    # In[137]:

    for i in range(len(data)):
        index = data.loc[i, 'Sentence'].find(':')
        if index >= 0:
            data.loc[i, 'Sentence'] = data.loc[i, 'Sentence'][index+2:]

    # In[138]:


    data = data.set_index('Speaker')


    # In[139]:
    index = f.index("/") + 1
    f = f[index:]
    filename = 'new_texts_for_tagging_speaker' + "/" + "Speaker Version " + f 
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w+', newline='') as file:
        data.to_csv(filename, sep = '\t')
# above line is error!!!1

    # In[ ]:




