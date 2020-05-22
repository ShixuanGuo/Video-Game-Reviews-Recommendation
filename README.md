# Video-Game-Reviews-Recommendation

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

4. **Packages**
  - NumPy
  - Pandas
  - nltk
  - sklearn
