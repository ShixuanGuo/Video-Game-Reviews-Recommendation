#!/usr/bin/env python
# coding: utf-8

# # Reviews preprocessing and text vectorization with TF-ID

# ## Concatenate comments for each video game

# In[126]:


df_comment_group_1=df_comment.groupby(['Title'])['Userscore','VADER Polarity','Comment'].agg({'Userscore':np.mean,                                                                                           'VADER Polarity':np.mean,                                                                           'Comment':lambda column: " ".join(column)})


# In[128]:


df_comment_group=df_comment_group_1.reset_index()


# ## Text Vectorization with Term Frequency - TF-IDF 

# In[131]:


from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer_TF_IDF = TfidfVectorizer(norm = None, smooth_idf = True,max_features=1000)


# In[132]:


NROM_comment = normalize_corpus(df_comment_group['Comment'])


# In[133]:


TF_IDF_matrix = vectorizer_TF_IDF.fit_transform(NROM_comment).toarray()


# In[134]:


TF_IDF_matrix_names = vectorizer_TF_IDF.get_feature_names() 
TF_IDF_matrix_table = pd.DataFrame(np.round(TF_IDF_matrix, 2), columns = TF_IDF_matrix_names)
TF_IDF_matrix_table.head()


# In[ ]:




