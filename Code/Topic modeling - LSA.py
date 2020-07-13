#!/usr/bin/env python
# coding: utf-8

# # Topic modeling - LSA

# In[1]:


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


# In[2]:


def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# In[3]:


n_comp = 35
no_top_words = 40

lsa = TruncatedSVD(n_components=n_comp)
lsa_tfidf_data = lsa.fit_transform(TF_IDF_matrix)


# In[4]:


display_topics(lsa,TF_IDF_matrix_names,no_top_words)


# In[5]:


# Display topics for LDA on TF-IDF Vectorizer (as comparison)
lda = LatentDirichletAllocation(n_components=n_comp)

lda_cv_data = lda.fit_transform(cv_data)
lda_tfidf_data = lda.fit_transform(tfidf_data)

display_topics(lda,tfidf_vectorizer.get_feature_names(),15)

