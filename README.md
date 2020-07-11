# Video-Game-Reviews-Recommendation

## Part 1 Project Introduction
1. **Problem description**  
  In the project, I developed a video game recommendation system based on a) reviews content and theme b) reviews’ sentiment. Given some key words, the system will recommend 10 video games matches the key words.
  
  
2. **Data description**  
  The dataset is called “Metacritic Video Game Comments”, which contains two separate .csv files: metacritic_game_info.csv and metacritic_game_user_comments.csv.  
  The dataset we used is **metacritic_game_user_comments.csv**, which includes 283960 video games comments for 2325 video games.  
  You can download data from https://www.kaggle.com/dahlia25/metacritic-video-game-comments.  

   **metacritic_game_user_comments.csv** contains six columns:
    - an index number from 0 to 283,982 represents a total of 283,983 comments;
    - video game title;
    - platform;
    - score given by the user who gave a review;
    - review text made by user;
    - username
  
  
3. **Implemented algorithms**
    - Lexicon-based sentiment analysis (VADER lexicon)
    - Reviews preprocessing and text vectorization with TF-IDF
    - Topic modeling (LSA, LDA)
    - K-nearest neighbors (KNN) recommendation model

## Part 2 Getting Started
1. **Install Packages**
    Mainly use:  
    - NumPy
    - Pandas
    - nltk
    - sklearn
    ```python
    import sys
    import numpy as np 
    import pandas as pd
    #!{sys.executable} -m pip install nltk
    import nltk

    import warnings
    warnings.simplefilter(action='ignore')

    !{sys.executable} -m pip install -U textblob
    
    %run ./Text_Normalization_Function.ipynb 
    ```

2. **Import dataset**  
Import dataset and reset index by using `df_comment = df_comment.rename(columns={"Unnamed: 0": "Number"}).set_index("Number")`.     
Keep only English comments.  
  ```python
     for index, row in df_comment.iterrows():
            l=row['Comment']
            if type(l) is not str:
                df_comment.drop(index, axis=0, inplace=True)
  ```
3. **Lexicon-based sentiment analysis (VADER lexicon)**  
    A. Find the true sentiment polarity according to user score: user score > 7.6 is positive.  
    ```python
    bin_Actual = []
    for i in df_comment['Userscore']:
        if i > 7.6:
            Pol = "positive"
        else:
            Pol = "negative"
        bin_Actual.append(Pol)
     ``` 
    B. Use VADER lexicon from the NLTK package `analyzer = SentimentIntensityAnalyzer()`.  
    Find VADER score of each comment using analyzer.  
      ```python
      Vader_scores=[]
      for comment in df_comment['Comment']:
          compound_score = analyzer.polarity_scores(comment)['compound']
          Vader_scores.append(compound_score)
      ```   
    Get VADER polarity given VADER score.  
      ```python
    VADER_polarity_test = [analyze_sentiment_vader_lexicon(comment, threshold=0.2) for comment in df_comment["Comment"]] 
      ```   
    C. Choose the best threshold parameter for our VADER model
    ```python
    thresholds = np.linspace(-1,1,1000)
    accuracy_rates = [try_threshold_for_accuracy(VADER_polarity_test_df["VADER Score"],threshold) for threshold in thresholds] 
    ```
    Try thresholds from -1 to 1 and plot the accuracy rates of sentiment polarity. The result shows best thresholds is -0.005, which means VADER score>-0.005 is positive.  
    ![Accuracy rate](https://github.com/ShixuanGuo/Video-Game-Reviews-Recommendation/blob/master/img/Accuracy%20rate.png)  
    D. Predict comment polarity using VADER model with the best threshold.   
    `VADER_polarity_test = [analyze_sentiment_vader_lexicon(comment, threshold= best_thresholds) for comment in df_comment["Comment"]]`  
    The true sentiment and VADER sentiment of top 5 comments:   
    ![Vader sentiment](https://github.com/ShixuanGuo/Video-Game-Reviews-Recommendation/blob/master/img/Vader%20sentiment.png)  

4. **Reviews preprocessing and text vectorization with TF-IDF**  
    A. In order to to extract the main features/or themes for each of the games, I concatenate comments for each video game
      ```python
      df_comment.groupby(['Title'])['Userscore','VADER Polarity','Comment'].agg({'Userscore':np.mean,\
                                                                                                 'VADER Polarity':np.mean,\
                                                                                 'Comment':lambda column: " ".join(column)})
      ```                                                                            
    B. Text Vectorization with TF-IDF: get the weights of text features  
      ```python
      from sklearn.feature_extraction.text import TfidfVectorizer 
      vectorizer_TF_IDF = TfidfVectorizer(norm = None, smooth_idf = True,max_features=1000)
      NROM_comment = normalize_corpus(df_comment_group['Comment'])
      TF_IDF_matrix = vectorizer_TF_IDF.fit_transform(NROM_comment).toarray()
      ```  
    I set the maximum number of text features as 1000, part of TF-IDF matrix is as follwing:  
    ![TF-IDF matrix](https://github.com/ShixuanGuo/Video-Game-Reviews-Recommendation/blob/master/img/TF-IDF%20table.png)  

5. **Topic modeling - LSA**  
In the topic modeling section, I decide to perform a LSA instead of LDA. Because the result of LDA does not perform as well as LSA. LSA learns latent topics by performing a matrix decomposition on the document-term matrix using singular value decomposition.  
Through LSA, I would be able to select the main 35 features/genres from video games and for each of the main feature, I select the 40 keywords.  
    ```python
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    n_comp = 35
    no_top_words = 40
    lsa = TruncatedSVD(n_components=n_comp)
    lsa_tfidf_data = lsa.fit_transform(TF_IDF_matrix)
    display_topics(lsa,TF_IDF_matrix_names,no_top_words)
    ```  
    Feature 0 and Feature 1 with their keywords:
    ![Topic with feature](https://github.com/ShixuanGuo/Video-Game-Reviews-Recommendation/blob/master/img/Topic%20with%20feature.png)  

6. **Recommendation System**  
    A. Import libraries need to be used from sklearn.   
      ```python
      from sklearn.model_selection import train_test_split
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn import metrics
      from sklearn.model_selection import KFold
      ```  
    B. Get recommendations using KNN.  
    I set k=10. The input is the features that you are searching and the output would be 10 related video games with sentiment review score as a supplemental information.  
    ```python
    X=lsa_tfidf_data
    y=df_comment_group['Title']
    result = get_recommendations(input, lsa, vectorizer_TF_IDF, X,y)
    ```   
    Using "warfare, multiplayer, duty, campaign" as input keywords, the recommendation result is:  
    ![Recommendation](https://github.com/ShixuanGuo/Video-Game-Reviews-Recommendation/blob/master/img/Recommendation.png)  

## Part 3 Files
1. Codes
  - Lexicon-based sentiment analysis
  - Reviews preprocessing and text vectorization
  - Topic modeling
  - Recommendation model
2. Support resources
  - Text_Normalization_Function (from WUSTL-B69 DAT 562)
