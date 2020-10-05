#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


client = pymongo.MongoClient(host="192.168.0.106", port=29197)
client["github"].authenticate("github", "git332", "github")
db = client["github"]


# In[3]:


lang_set = [x["name"] for x in db.project_lang_30.find()]
npm_tag_set = ["npm_"+x["name"] for x in db.npm_tag_stem_top100.find()]
pkg_tag_set = ["pkg_"+x["name"] for x in db.composer_tag_stem_top100.find()]
pypi_tag_set = ["pypi_"+x["name"] for x in db.pypi_tag_stem_top100.find()]


# In[4]:


train_col = ["file_npm", "file_pypi", "file_composer"]
train_col.extend(lang_set)
train_col.extend(npm_tag_set)
train_col.extend(pkg_tag_set)
train_col.extend(pypi_tag_set)
len(train_col)


# In[147]:


lda_dim = 256


# In[5]:


tag_set = [x["name"] for x in db.project_tag_more_than_100.find()]
len(tag_set)


# In[148]:


lda = models.ldamodel.LdaModel.load('../model/readme_lda_{}.model'.format(lda_dim))


# In[111]:


data_raw = pd.read_csv("../data/data_raw_tagonehot.csv", index_col=0)
data_raw.head(4)


# In[112]:


data_tag = np.array(data_raw[tag_set]).astype(np.int64)


# In[113]:


data_minus_readme = data_raw[train_col]
data_minus_readme.shape


# In[114]:


readme_list = list(data_raw.file_readme)
corpus = []

for post in readme_list:
    
    corpus.append([word for word in str(post).strip().lower().split()])

len(readme_list)


# In[115]:


dictionary = Dictionary(corpus)
doc_bow = dictionary.doc2bow(corpus[0])
lda_list = lda[doc_bow]


# In[116]:


corpus_lda = [lda[dictionary.doc2bow(doc)] for doc in corpus]


# In[149]:


data_matrix = []

for post in corpus_lda:
    arr = np.zeros(lda_dim)
    for elem in post:
        arr[elem[0]] = elem[1]
    
    data_matrix.append(arr)


# In[118]:


data_x_np = np.array(data_matrix)
data_x_np.shape


# In[119]:


data_full_feature = np.concatenate((data_x_np,data_minus_readme),axis=1)


# In[150]:


data_full_feature.shape[1]


# In[121]:


len(data_x_np[1])


# In[122]:


len(tag_set)


# In[123]:


train_x, test_x, train_y, test_y = train_test_split(data_full_feature, data_tag, train_size=0.9, test_size=0.1, random_state=1)


# In[124]:


len(train_x), len(test_x)


# In[164]:


import keras as K
input_dim = data_full_feature.shape[1]
lr = 0.0001
init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam(lr=lr)
model = K.models.Sequential()


model.add(K.layers.Dense(units=600, input_dim=input_dim, kernel_initializer=init, activation='relu'))
# model.add(K.layers.Dense(units=600, kernel_initializer=init, activation='relu'))
# model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(units=400, kernel_initializer=init, activation='relu'))
# model.add(K.layers.BatchNormalization())

model.add(K.layers.Dense(units=200, kernel_initializer=init, activation='relu'))
# model.add(K.layers.BatchNormalization())

model.add(K.layers.Dense(units=116, kernel_initializer=init, activation='relu'))
model.compile(loss='mean_squared_error', optimizer=simple_adam, metrics=['top_k_categorical_accuracy', 'categorical_accuracy'])


# In[165]:


b_size = 128
max_epochs = 1000
print("Starting training ")
h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")


# In[166]:


p = model.predict(test_x)


# In[167]:


p[0],test_y[0]


# In[168]:


# 取出分数不为0的项
p_list = []
for k in p:
    p_tag = {}
    for i, score in enumerate(k.tolist()):
        if(score!=0):
            p_tag[score] = i
    p_list.append(p_tag)


# In[169]:


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


# In[170]:


p_list_top5 = []
for i in range(len(p_list)):
    p_list_top5.append([p_list[i][k] for k in sorted(p_list[i].keys())][-5:])
    
p_list_top10 = []
for i in range(len(p_list)):
    p_list_top10.append([p_list[i][k] for k in sorted(p_list[i].keys())][-10:])
    
p_list_top116 = []
for i in range(len(p_list)):
    p_list_top116.append([p_list[i][k] for k in sorted(p_list[i].keys())][-116:])


# In[171]:


np.array([0,0,2]).nonzero()


# In[172]:


true_tag_real = []
for row in test_y:
    true_tag_real.append(row.nonzero()[0])


# In[173]:


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


print("lr: {} b_size: {} max_epochs: {} lda_dim: {} ".format(lr, b_size, max_epochs, lda_dim))

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

model.save("../model/lr-{}-b_size-{}-max_epochs-{}-lda_dim-{}.h5 ".format(lr, b_size, max_epochs, lda_dim))