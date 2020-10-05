#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pandas as pd
from gensim import corpora, models
from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import keras as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import tensorflow as tf
import pymongo

import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import string, re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate,MaxPooling1D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
import keras as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers


# In[153]:


embed_size = 300 # how big is each word vector
max_features = 200000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 300 


# In[75]:


client = pymongo.MongoClient(host="192.168.0.106", port=29197)
client["github"].authenticate("github", "git332", "github")
db = client["github"]


# In[76]:


lang_set = [x["name"] for x in db.project_lang_30.find()]
npm_tag_set = ["npm_"+x["name"] for x in db.npm_tag_stem_top100.find()]
pkg_tag_set = ["pkg_"+x["name"] for x in db.composer_tag_stem_top100.find()]
pypi_tag_set = ["pypi_"+x["name"] for x in db.pypi_tag_stem_top100.find()]


# In[77]:


train_col = ["file_npm", "file_pypi", "file_composer"]
train_col.extend(lang_set)
train_col.extend(npm_tag_set)
train_col.extend(pkg_tag_set)
train_col.extend(pypi_tag_set)
len(train_col)


# In[78]:


tag_set = [x["name"] for x in db.project_tag_more_than_100.find()]
len(tag_set)


# In[79]:


lda = models.ldamodel.LdaModel.load('../model/readme_lda_256.model')


# In[80]:


data_raw = pd.read_csv("../data/data_raw.csv", index_col=0)
data_raw.head(4)


# In[81]:


data_tag = np.array(data_raw[tag_set]).astype(np.int64)


# In[150]:


readme_x = data_raw["file_readme"].fillna("_##_").values


# In[154]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(readme_x))
readme_x_tz = tokenizer.texts_to_sequences(readme_x)


# In[156]:


## Pad the sentences 
train_X_seq = pad_sequences(readme_x_tz, maxlen=maxlen)
train_X_seq.shape


# In[155]:


vocab = tokenizer.word_index


# In[82]:


data_minus_readme = data_raw[train_col]
data_minus_readme.shape


# In[159]:


train_x, test_x, train_y, test_y = train_test_split(train_X_seq, data_tag, train_size=0.75, test_size=0.25, random_state=2019)


# In[167]:


train_x.shape, test_x.shape


# In[187]:


# model define



# model = K.models.Sequential()
# model.add(Embedding(len(vocab), 100, input_length=300))
# model.add(Flatten())
# model.add(K.layers.Dense(units=200, input_dim=100, kernel_initializer=init, activation='relu'))
# model.add(K.layers.Dense(units=200, kernel_initializer=init, activation='relu'))
# model.add(K.layers.Dense(units=200, kernel_initializer=init, activation='relu'))
# model.add(K.layers.Dense(units=1, kernel_initializer=init, activation='relu'))
# model.compile(loss='mean_squared_error', optimizer=simple_adam, metrics=['categorical_accuracy'])

def BuildTextCNN(maxlen=200, max_features=20000, embed_size=64):
    comment_seq = Input(shape=[maxlen], name='x_seq')
    emb_comment = Embedding(max_features, embed_size)(comment_seq)
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)
    output = Dense(200, activation='relu')(out)
    output = Dense(units=116, activation='relu')(output)
    model = Model([comment_seq], output)
    return model


# In[188]:

simple_adam = K.optimizers.Adam(lr=0.0001)
init = K.initializers.glorot_uniform(seed=1)
textcnn = BuildTextCNN(maxlen=maxlen,max_features=len(vocab))
textcnn.compile(loss='mean_squared_error', optimizer=simple_adam, metrics=['categorical_accuracy'])


# In[189]:


b_size = 32
max_epochs = 50
print("Starting training ")
h = textcnn.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")


# In[190]:


textcnn.summary()


# In[191]:


p = textcnn.predict(test_x)


# In[192]:


p[0],test_y[0]


# In[193]:


# 取出分数不为0的项
p_list = []
for k in p:
    p_tag = {}
    for i, score in enumerate(k.tolist()):
        if(score!=0):
            p_tag[score] = i
    p_list.append(p_tag)


# In[194]:


def recall(true_tag, pre_tag):
    t = 0
    for i in pre_tag:
        if(i in true_tag):
            t += 1
    return t/len(true_tag)

def precision(true_tag, pre_tag):
    t = 0
    for i in pre_tag:
        if(i in true_tag):
            t += 1
    if len(pre_tag) == 0:
        return 1
    else:
        return t/len(pre_tag)


# In[195]:


p_list_top5 = []
for i in range(len(p_list)):
    p_list_top5.append([p_list[i][k] for k in sorted(p_list[i].keys())][-5:])
    
p_list_top10 = []
for i in range(len(p_list)):
    p_list_top10.append([p_list[i][k] for k in sorted(p_list[i].keys())][-10:])
    
p_list_top116 = []
for i in range(len(p_list)):
    p_list_top116.append([p_list[i][k] for k in sorted(p_list[i].keys())][-116:])


# In[197]:


true_tag_real = []
for row in test_y:
    true_tag_real.append(row.nonzero()[0])


# In[198]:


recal_at_116_count = 0
recal_at_10_count = 0
recal_at_5_count = 0
precision_at_116_count = 0
precision_at_10_count = 0
precision_at_5_count = 0

for i in range(len(true_tag_real)):
    recal_at_10_count += recall(true_tag_real[i], p_list_top10[i])


for i in range(len(true_tag_real)):
    recal_at_5_count += recall(true_tag_real[i], p_list_top5[i])
    
for i in range(len(true_tag_real)):
    recal_at_116_count += recall(true_tag_real[i], p_list_top116[i])
    
for i in range(len(true_tag_real)):
    precision_at_10_count += precision(true_tag_real[i], p_list_top10[i])    

for i in range(len(true_tag_real)):
    precision_at_5_count += precision(true_tag_real[i], p_list_top5[i])
    
for i in range(len(true_tag_real)):
    precision_at_116_count += precision(true_tag_real[i], p_list_top116[i])
    
recall_at_116 = recal_at_116_count/len(true_tag_real)
recall_at_10 = recal_at_10_count/len(true_tag_real)
recall_at_5 = recal_at_5_count/len(true_tag_real)
precision_at_116 = precision_at_116_count/len(true_tag_real)
precision_at_10 = precision_at_10_count/len(true_tag_real)
precision_at_5 = precision_at_5_count/len(true_tag_real)

model.summary()
print()
print("====================================")
print("Recall@116: {}".format(recall_at_116))
print("Precision@116: {}".format(precision_at_116_count/len(true_tag_real)))
print("F1-score@116: {}".format((2*recall_at_116*precision_at_10)/(recall_at_116+precision_at_10)))
print()
print("Recall@10: {}".format(recall_at_10))
print("Precision@10: {}".format(precision_at_10_count/len(true_tag_real)))
print("F1-score@10: {}".format((2*recall_at_10*precision_at_10)/(recall_at_10+precision_at_10)))
print()
print("Recall@5: {}".format(recall_at_5))
print("Precision@5: {}".format(precision_at_5_count/len(true_tag_real)))
print("F1-score@5: {}".format((2*recall_at_5*precision_at_5)/(recall_at_5+precision_at_5)))
print("====================================")


# In[200]:


import datetime
textcnn.save("../model/textcnn-{}.h5".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

