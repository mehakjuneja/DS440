#!/usr/bin/env python
# coding: utf-8

# #**Group 4 Code Implementation**#
# #Movie Reccomendation System#
# ###Mehak Juneja, Saketh Gudiseva, Sooraj Narayanan

# https://github.com/sanjayjaras/sanjayjaras.github.io/blob/master/Projects/movies-recommendation/movie-recommendation-final.ipynb

# In[ ]:


# conda install -c conda-forge scikit-surprise


# #Import Libraries

# In[38]:


import numpy as np 
import pandas as pd 
import surprise
import matplotlib.pyplot as plt
from matplotlib import __version__ as mpv
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sparse
from surprise.model_selection import train_test_split

from surprise import SVD
from surprise import accuracy
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import Reader
from collections import defaultdict
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# In[39]:


# from google.colab import drive
# drive.mount('/content/drive', force_remount = True)

#https://drive.google.com/drive/folders/18xDVcDln5ZU-iwP2MEUKajbGi769E7HZ?usp=share_link


# #Configurations

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn-darkgrid")
random_state = 17
np.random.seed(random_state)
import warnings
warnings.filterwarnings('ignore')


# In[41]:


# rating=pd.read_csv("dataset/ratings.csv")
# link=pd.read_csv("dataset/links.csv")
# movies=pd.read_csv("dataset/movies.csv")

# tags=pd.read_csv("dataset/tags.csv")
# genome_scores=pd.read_csv("dataset/genome-scores.csv")
# genome_tags=pd.read_csv("dataset/genome-tags.csv")

# metadata = pd.read_csv("dataset/movies_metadata.csv")


# #Load Datasets

# In[42]:


#!unzip "/content/drive/My Drive/DS 440/movies_metadata.csv.zip"


# In[43]:


#rating = pd.read_csv("/content/drive/My Drive/DS 440/data/ratings.csv") #import data
rating = pd.read_csv("https://github.com/mehakjuneja/DS440/releases/download/ratings.large/ratings.csv")
#link = pd.read_csv("/content/drive/My Drive/DS 440/data/links.csv") #import data
link = pd.read_csv("https://raw.githubusercontent.com/mehakjuneja/DS440/main/links.csv") 

#movies = pd.read_csv("/content/drive/My Drive/DS 440/data/movies.csv") #import data
c

#tags=pd.read_csv("/content/drive/My Drive/DS 440/data/tags.csv")
tags=pd.read_csv("https://github.com/mehakjuneja/DS440/releases/download/tags.large/tags.csv")
#genome_scores=pd.read_csv("/content/drive/My Drive/DS 440/data/genome-scores.csv")
genome_scores = pd.read_csv("https://github.com/mehakjuneja/DS440/releases/download/scores.large/genome-scores.csv")

#genome_tags=pd.read_csv("/content/drive/My Drive/DS 440/data/genome-tags.csv")
genome_tags=pd.read_csv("https://raw.githubusercontent.com/mehakjuneja/DS440/main/genome-tags.csv")

#metadata = pd.read_csv("movies_metadata.csv")
metadata = pd.read_csv("https://github.com/mehakjuneja/DS440/releases/download/metadata.large/movies_metadata.csv")


# ##Movie dataset

# In[44]:


movies.info()


# In[45]:


movies.head(5)


# ###Rating dataset
# 

# In[ ]:


rating.info()


# In[ ]:


rating.head(5)


# ##Link dataset holding relational keys to IMDb and TMDB datasets

# In[ ]:


link.info()


# In[ ]:


link.head(5)


# ###Scoring Dataset

# In[ ]:


genome_scores.info()


# In[ ]:


genome_scores.head(5)


# ###Genome Tags Dataset

# In[ ]:


genome_tags.info()


# In[ ]:


genome_tags.head(5)


# ###Tags Dataset

# In[ ]:


tags.info()


# In[ ]:


tags.head(5)


# ###Metadata Dataset

# In[ ]:


metadata.info()


# In[ ]:


metadata.head(5)


# ##EDA

# ###Extract Movie Year from Title

# In[46]:


movies["year"] = movies.title.str.extract('(\(\d{4}\))')
movies.year = movies.year.str.extract('(\d+)')
movies.year = pd.to_numeric(movies.year)


# ###Movies by Decade

# In[5]:


fig, ax = plt.subplots(figsize=(20, 8))
p1=sns.histplot(data=movies, x='year', ax=ax, binwidth=10)
plt.title('Movie count by decade')
plt.xticks(rotation=60)
plt.show()


# ###Split genre column to dummy columns

# In[47]:


genres = movies.genres.str.get_dummies().add_prefix('g_')
movies = pd.concat([movies, genres], axis=1)


# In[ ]:


movies.head()


# ###Movies By Genre

# In[48]:


g_cols = [ col for col in movies.columns if col.startswith("g_")] 


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 8))
p1=sns.barplot(x=g_cols, y=movies[g_cols].sum(), ax=ax)
plt.title('Movies count by Genre')
plt.xticks(rotation=60)
plt.show()


# ### Movies by decade and genre

# In[49]:


by_decade_genres= movies[g_cols].groupby(np.floor(movies.year/10)*10).sum()
by_decade_genres = by_decade_genres.transpose()
for col in by_decade_genres.columns:
    by_decade_genres[col] = by_decade_genres[col]/by_decade_genres[col].sum()

by_decade_genres


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 10))
p1=plt.stackplot(by_decade_genres.columns, by_decade_genres, labels=by_decade_genres.index)
plt.title('Movies Genre by Decade')
plt.xticks(rotation=60)
plt.legend(loc="upper left")
plt.show()


# ###Add year column to metadata from release data

# In[50]:


metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[51]:


metadata


# ###Utility functions

# In[52]:


def to_int(x):
    try:
        return int(x)
    except:
        return np.nan

def get_first_index(idx):
    if isinstance(idx, list) or isinstance(idx, pd.Series):
        idx = idx[0]
    return idx    


# ###Convert Id column to int from object to connect with links

# In[53]:


metadata['id'] = metadata['id'].apply(to_int)


# ##Model Creation

# ###Simple Recommendation model using weighted-rating

# ####Function to return sorted list of movies by weighted rating

# #####Following formula is used to calculated weighted rating Weighted Rating (WR) = (v/(v+m) * R ) + (m/(v+m)*C)
#  
#  
# 

# In[54]:


def get_top_weighted_rating(df, number_of_records=200, percentile=0.85):
    non_null_vote_counts = df[df['vote_count'].notnull()]['vote_count']
    non_null_vote_avgs = df[df['vote_average'].notnull()]['vote_average']
    mean_votes = non_null_vote_avgs.mean()
    min_votes_req = non_null_vote_counts.quantile(percentile)

    selected = df[(df['vote_count'] >= min_votes_req) & (
        df['vote_count'].notnull()) & (df['vote_average'].notnull())]
    selected = selected[['title', 'year',
                         'vote_count', 'vote_average', 'popularity', 'id']]
    selected['weighted_rating'] = selected.apply(lambda x: (
        x.vote_count / (x.vote_count + min_votes_req) * x['vote_average']) + (min_votes_req/(min_votes_req + x.vote_count) * mean_votes), axis=1)

    selected = selected.sort_values(
        'weighted_rating', ascending=False).head(number_of_records)
    return selected


# ####Function to create top movie charts for all movies and by genre

# In[55]:


def build_top_movie_chart(dataframe, genre=None, percentile=0.85, no_of_movies=200):
    if genre is None:
        df = dataframe
    else:
        df = stack_df_by_genre(dataframe)
        df = df[df['genre'] == genre]
    selected = get_top_weighted_rating(df, no_of_movies, percentile)
    return selected

def stack_df_by_genre(dataframe):
    metadata_temp = dataframe.copy()
    metadata_temp['genres'] = metadata_temp['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df = metadata_temp.apply(lambda x: pd.Series(
        x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    df.name = 'genre'
    df = metadata_temp.drop('genres', axis=1).join(df)
    return df

def replace_genre_json_with_list(dataframe, fieldName):
    metadata_temp = dataframe.copy()
    metadata_temp[fieldName] = metadata_temp[fieldName].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    metadata_temp[fieldName] = metadata_temp[fieldName].apply(lambda x: ','.join(map(str, x)))
    return metadata_temp    


# ####Top 10 movies by weighted rating

# In[56]:


build_top_movie_chart(metadata, percentile=0.95, no_of_movies=10)


# ####Top 10 Crime movies

# In[ ]:


build_top_movie_chart(metadata, genre="Crime", percentile=0.90, no_of_movies=10)


# ####Top 10 Drama movies

# In[ ]:


build_top_movie_chart(metadata, genre="Drama", percentile=0.90, no_of_movies=10)


# ###**Content Based Recommendation**

# #####Using genres, spoken_languages, tagline, and overview from metadata dataset to create Content based recommendation

# ####Create new column desc by concatenating 4 column contents spoken_languages, tagline, and overview from metadata dataset

# In[58]:


metadata


# In[61]:


metadata_for_cont = metadata.copy()
metadata_for_cont = replace_genre_json_with_list(
    metadata_for_cont, "genres")
metadata_for_cont = replace_genre_json_with_list(
    metadata_for_cont, "spoken_languages")

metadata_for_cont = replace_genre_json_with_list(
    metadata_for_cont, "belongs_to_collection")

metadata_for_cont["genres"] = metadata_for_cont.genres.fillna("")
metadata_for_cont["spoken_languages"] = metadata_for_cont.spoken_languages.fillna(
    "")
metadata_for_cont["tagline"] = metadata_for_cont.tagline.fillna("")
metadata_for_cont["belongs_to_collection"] = metadata_for_cont.belongs_to_collection.fillna("")
metadata_for_cont["desc"] = metadata_for_cont.genres +     metadata_for_cont.spoken_languages +     metadata_for_cont.overview + metadata_for_cont.tagline +metadata_for_cont.belongs_to_collection
metadata_for_cont["desc"] = metadata_for_cont.desc.fillna("")

metadata_for_cont = metadata_for_cont[metadata_for_cont.id.isin(link.tmdbId)]


# In[ ]:





# ####Content that will be used for Content-Based recommendation
# 

# In[19]:


metadata_for_cont.desc


# ####Create n-gram and vectorize for each movie

# In[66]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                     min_df=0, stop_words='english')
metadata_transformed = tf.fit_transform(metadata_for_cont.desc)


# In[67]:


metadata_transformed.shape


# In[68]:


#metadata_transformed


# ####Calculate cosine simalirity between all movies by using words from desc column

# 

# In[69]:


cosine_sim = linear_kernel(metadata_transformed, metadata_transformed)


# In[70]:


metadata_for_cont = metadata_for_cont.reset_index()
titles = metadata_for_cont['title']
indices = pd.Series(metadata_for_cont.index, index=metadata_for_cont['title'])


# ####Function to sort movies by simalarity and return(default 30) similar movies of the movie passed as parameter

# In[71]:


def get_recommendations(title, no_of_movies=30):
    idx = get_first_index(indices[title])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:no_of_movies+1]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# ####Similar movies to "The Apartment" movie

# In[72]:


get_recommendations('The Apartment', 10)


# ###**Hybrid model combining weighted rating + content-based model**

# ####Function to calculate Similarity by considering simalarity scores from cosine similarity followed by weighted ratings

# In[73]:


def cosine_sim_plus_weighted_rating(title, no_of_movies, quantile=0.60):
    idx = get_first_index(indices[title])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:no_of_movies+20]
    movie_indices = [i[0] for i in sim_scores]

    movs = metadata_for_cont.iloc[movie_indices][[
        'title', 'vote_count', 'vote_average', 'year', 'popularity', 'id']]
    selected = get_top_weighted_rating(movs, no_of_movies, 0.60)
    return selected


# Similar movies to "The Family" movie by cosine similarity and weighted rating

# In[74]:


cosine_sim_plus_weighted_rating('The Family', 10, 0.80)


# Similar movies to "The Apartment" movie by cosine similarity and weighted rating

# In[75]:


cosine_sim_plus_weighted_rating('The Apartment', 10)


# In[29]:


cosine_sim_plus_weighted_rating('The Apartment', 10)


# ###**Collaborative filtering model by using user ratings and finding similar users**

# ####Movies wathced by User with id 5

# In[76]:


def user_watched(user):
    watched_movies = rating[rating.userId == user]
    return pd.DataFrame({"title": movies.title.iloc[watched_movies.movieId], "genres": movies.genres.iloc[watched_movies.movieId], "rating": watched_movies.rating.values})


user_watched(5)


# ####Prepare dataset for rating model

# #####Conver ratings to int8 from float

# In[77]:


rating["rating"] = rating.rating.astype("int8")


# In[78]:


reader = Reader(rating_scale=(1, 5))
train_set_for_grid_search = rating[:100000]
train_set_for_grid_search = Dataset.load_from_df(train_set_for_grid_search[['userId', 'movieId', 'rating']], reader)


# ####Grid search to fine-tune hyperparameters and model selection
# 

# In[80]:


param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=10, joblib_verbose=2 )

gs.fit(train_set_for_grid_search)

print("Model Name", "SVD")
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])


param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6], 'k': [50, 100, 200]}
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=10, joblib_verbose=2 )

gs.fit(train_set_for_grid_search)


print("Model Name", "KNN")
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])


# ####build model with full dataset

# #####Though the Knn model is better as per grid search reasults I am going with SVD as my hardware is not supporting KNN

# ####prepare dataset for surprise models and split into train and test dataset

# In[34]:


data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)


# In[ ]:


#knn = KNNWithMeans(k=50, measures=['rmse', 'mae'],
#                   cv=3, n_jobs=10, joblib_verbose=2, pre_dispatch=1)
#knn.fit(trainset)


# ####Train SVD model on training dataset to calculate accuracy
# 

# In[35]:


svd = SVD(verbose=True, n_epochs=10,  lr_all=0.005, reg_all=0.4)
svd.fit(trainset)


# ####Accuracy on test dataset

# In[36]:


predictions = svd.test(testset)
# Compute and print Root Mean Squared Error
accuracy.rmse(predictions, verbose=True)


# In[ ]:


svd.fit(data.build_full_trainset())


# In[ ]:


svd.predict(uid=5, iid=100)


# ###**Hybrid model combining weighted rating+content-based cosine simalarity+user based rating based collaborative filtering type**

# ####Combine movie dataset with tmdb dataset

# In[ ]:


movieId_tmdbId = link.copy()
movieId_tmdbId = movieId_tmdbId[["movieId", "tmdbId"]]
movieId_tmdbId.columns = ['movieId', 'id']
movieId_tmdbId = movieId_tmdbId.merge(metadata_for_cont[['title', 'id']], on='id').set_index('title')
movieId_tmdbId


# ####function to create Hybrid model

# In[ ]:


tmdbId_index_movie = movieId_tmdbId.set_index('id')

def hybrid_model_cosine_weighted_rate_svd(userId, title, filter_on_weighted_rate=True):
    idx = get_first_index(indices[title])
    tmdbId = movieId_tmdbId.loc[title]['id']
    #print(idx)
    movie_id = movieId_tmdbId.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:40]
    movie_indices = [i[0] for i in sim_scores]

    mov = metadata_for_cont.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id', 'popularity']]
    if filter_on_weighted_rate:
        mov = get_top_weighted_rating(mov, 200, 0.30)

    mov['est'] = mov['id'].apply(lambda x: svd.predict(userId, tmdbId_index_movie.loc[x]['movieId']).est)
    mov = mov.sort_values('est', ascending=False)
        
    return mov.head(10)


# ####Recommended movies for user 5, who watched movie Jumanji and filter_on_weighted_rate=False

# In[ ]:


hybrid_model_cosine_weighted_rate_svd(5, 'Jumanji', filter_on_weighted_rate=False)


# ####Recommended movies for user 5, who watched movie Jumanji and filter_on_weighted_rate=True

# In[ ]:


hybrid_model_cosine_weighted_rate_svd(5, 'Jumanji', filter_on_weighted_rate=True)


# ####Recommended movies for user 5, who watched movie Kung Fu Panda and filter_on_weighted_rate=False

# In[ ]:


hybrid_model_cosine_weighted_rate_svd(5, 'Kung Fu Panda', filter_on_weighted_rate=False)


# ####Recommended movies for user 5, who watched movie Kung Fu Panda and filter_on_weighted_rate=True

# In[ ]:


hybrid_model_cosine_weighted_rate_svd(5, 'Kung Fu Panda', filter_on_weighted_rate=True)


# ####Recommended movies for user 555, who watched movie Kung Fu Panda and filter_on_weighted_rate=False

# In[ ]:


hybrid_model_cosine_weighted_rate_svd(555, 'Kung Fu Panda', filter_on_weighted_rate=False)


# ####Recommended movies for user 555, who watched movie Kung Fu Panda and filter_on_weighted_rate=True

# In[ ]:


hybrid_model_cosine_weighted_rate_svd(555, 'Kung Fu Panda', filter_on_weighted_rate=True)


# 
