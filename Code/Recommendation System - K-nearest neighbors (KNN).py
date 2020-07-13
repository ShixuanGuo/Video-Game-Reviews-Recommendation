#!/usr/bin/env python
# coding: utf-8

# # Recommendation System - K-nearest neighbors (KNN)

# In[140]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold


# In[156]:


def get_recommendations(first_article, model, vectorizer, X,y):
    new_vec = model.transform(vectorizer.transform([first_article]))
    nn = KNeighborsClassifier(n_neighbors=10, metric='cosine', algorithm='brute').fit(X,y)
    results = nn.kneighbors(new_vec)
    return results[1][0]


# In[160]:


X=lsa_tfidf_data
y=df_comment_group['Title']
inputs="warfare, multiplayer, duty, campaign"
result = get_recommendations(inputs, lsa, vectorizer_TF_IDF, X,y)
result


# In[161]:


# show 10 related video games with sentiment review score
for r in result:
    game = df_comment_group.Title[r]
    vader_polarity = df_comment_group["VADER Polarity"][r]
    print(f'Recommend games name is {game}. \nThe sentiment score of is {round(vader_polarity,2)}.')


# In[ ]:




