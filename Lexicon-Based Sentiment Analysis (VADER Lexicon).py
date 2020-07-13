#!/usr/bin/env python
# coding: utf-8

# ## Lexicon-Based Sentiment Analysis (VADER Lexicon)

# In[4]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[106]:


# Binary Actual Userscore Polarity
bin_Actual = []
for i in df_comment['Userscore']:
    if i > 7.6:
        Pol = "positive"
    else:
        Pol = "negative"
    bin_Actual.append(Pol)

df_comment["Bin Actual Userscore Polarity"]=bin_Actual
df_comment.reset_index()


# In[107]:


#VADER
Vader_scores=[]
for comment in df_comment['Comment']:
    compound_score = analyzer.polarity_scores(comment)['compound']
    Vader_scores.append(compound_score)
    

df_comment['Vader Score'] = Vader_scores


# In[108]:


# given the vader score, get polarity
def analyze_sentiment_vader_lexicon(review, threshold = 0.1, verbose = False):
    scores = analyzer.polarity_scores(review)  
    binary_sentiment = 'positive' if scores['compound'] >= threshold else 'negative'
    if verbose:                             
        print('VADER Polarity (Binary):', binary_sentiment)
        print('VADER Score:', round(scores['compound'], 2))
    return binary_sentiment,scores['compound']  


# In[109]:


VADER_polarity_test = [analyze_sentiment_vader_lexicon(comment, threshold=0.2) for comment in df_comment["Comment"]]
VADER_polarity_test_df = pd.DataFrame(VADER_polarity_test, columns = ['VADER Polarity','VADER Score'])
from sklearn import metrics
print('Accuracy Rate:', np.round(metrics.accuracy_score(df_comment["Bin Actual Userscore Polarity"], 
                                 VADER_polarity_test_df["VADER Polarity"]), 3),"\n")


# ## Find-tune the best threshold parameter

# In[111]:


def try_threshold_for_accuracy(sentiment_scores, threshold_for_pos):
    VADER_binary_polarity = ['positive' if s >= threshold_for_pos else 'negative' for s in list(sentiment_scores)]
    accuracy = metrics.accuracy_score(df_comment["Bin Actual Userscore Polarity"], VADER_binary_polarity)
    return(accuracy) 


# In[118]:


# show prediction accuracy as a table
pd.crosstab(pd.Series(df_comment["Bin Actual Userscore Polarity"]), 
            pd.Series(VADER_polarity_test_df['VADER Polarity']), 
            rownames = ['True:'], 
            colnames = ['Predicted:'], 
            margins = True)


# In[115]:


# plot the threshold and accuracy rate
import matplotlib.pyplot as plt
thresholds = np.linspace(-1,1,1000)
accuracy_rates = [try_threshold_for_accuracy(VADER_polarity_test_df["VADER Score"],threshold) for threshold in thresholds]

plt.plot(thresholds, accuracy_rates)
plt.xlabel("Threshold score for positive (binary) sentiment polarity")
plt.ylabel("Accuracy rate")
plt.title("Accuracy Rate of Sentiment Polarity Prediction \n as a Function of Threshold for VADER Scores \n")
plt.show()


# In[116]:


best_thresholds = thresholds[accuracy_rates.index(max(accuracy_rates))] 

VADER_polarity_test = [analyze_sentiment_vader_lexicon(comment, threshold= best_thresholds) for comment in df_comment["Comment"]]
VADER_polarity_test_df = pd.DataFrame(VADER_polarity_test, columns = ['VADER Polarity','VADER Score'])
df_comment["VADER Polarity"] = VADER_polarity_test_df['VADER Polarity']


# In[125]:


df_comment["VADER Polarity"]=df_comment["VADER Polarity"].apply(lambda x: 1 if x is "positive" else 0)

