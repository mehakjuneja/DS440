import streamlit as st

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

def loading_message_movie(func):
    def wrapper(*args, **kwargs):
        loading_text = "Hang tight, while we find the movies for you..."
        with st.spinner(loading_text):
            result = func(*args, **kwargs)
        st.success('We hope you enjoy these movies!')
        return result
    return wrapper


# Data URLS
rating = ("https://github.com/mehakjuneja/DS440/releases/download/ratings.large/ratings.csv")
link = ("https://raw.githubusercontent.com/mehakjuneja/DS440/main/links.csv") 
movies=("https://raw.githubusercontent.com/mehakjuneja/DS440/main/movies.csv")
tags=("https://github.com/mehakjuneja/DS440/releases/download/tags.large/tags.csv")
genome_scores = ("https://github.com/mehakjuneja/DS440/releases/download/scores.large/genome-scores.csv")
genome_tags=("https://raw.githubusercontent.com/mehakjuneja/DS440/main/genome-tags.csv")
metadata = ("https://github.com/mehakjuneja/DS440/releases/download/metadata.large/movies_metadata.csv")

@st.cache_data
def load_data(dt_url, nrows=1000):
    name= str((movies.split('.')[-2]).split('/')[-1])
    data = pd.read_csv(dt_url)
    #data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    #st.subheader(f'{name} data')
    #st.write(data)
    return data


@st.cache_data
def decade_graph(movies):
    m_decade = load_data(movies)
    m_decade["year"] = m_decade.title.str.extract('(\(\d{4}\))')
    m_decade.year = m_decade.year.str.extract('(\d+)')
    m_decade.year = pd.to_numeric(m_decade.year)
    st.subheader('Movies By Decade')
    fig, ax = plt.subplots(figsize=(20, 8))
    p1=sns.histplot(data=m_decade, x='year', ax=ax, binwidth=10, kde=True)

    plt.xticks(rotation=60)
    st.pyplot(fig)
    
    #return movies

@st.cache_data
def genre_graph(movies):
    m_genre = load_data(movies)
    genres = m_genre.genres.str.get_dummies().add_prefix('g_')
    g_data = pd.concat([m_genre, genres], axis=1)
    g_cols = [ col for col in g_data.columns if col.startswith("g_")] 
    fig, ax = plt.subplots(figsize=(20, 8))
    p1=sns.barplot(x=g_cols, y=g_data[g_cols].sum(), ax=ax)
    st.subheader('Movies count by Genre')
    plt.xticks(rotation=60)
    st.pyplot(fig)

@st.cache_data
def genre_decade_graph(movies):
    m_genre = load_data(movies)
    genres = m_genre.genres.str.get_dummies().add_prefix('g_')
    g_data = pd.concat([m_genre, genres], axis=1)
    g_data["year"] = g_data.title.str.extract('(\(\d{4}\))')
    g_data.year = g_data.year.str.extract('(\d+)')
    g_data.year = pd.to_numeric(g_data.year)
    g_cols = [ col for col in g_data.columns if col.startswith("g_")] 
    by_decade_genres= g_data[g_cols].groupby(np.floor(g_data.year/10)*10).sum()
    by_decade_genres = by_decade_genres.transpose()
    for col in by_decade_genres.columns:
        by_decade_genres[col] = by_decade_genres[col]/by_decade_genres[col].sum()
    fig, ax = plt.subplots(figsize=(20, 10))
    p1=plt.stackplot(by_decade_genres.columns, by_decade_genres, labels=by_decade_genres.index)
    st.subheader('Movies Genre by Decade')
    plt.xticks(rotation=60)
    plt.legend(loc="upper left")
    st.pyplot(fig)

@st.cache_data
def to_int(x):
    try:
        return int(x)
    except:
        return np.nan
@st.cache_data
def get_first_index(idx):
    if isinstance(idx, list) or isinstance(idx, pd.Series):
        idx = idx[0]
    return idx 

# load_data(rating)
# decade_graph(movies)
# genre_graph(movies)
# genre_decade_graph(movies)

md_df=load_data(metadata)
md_df['year'] = pd.to_datetime(md_df['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

md_df['id'] = md_df['id'].apply(to_int)

#st.write(md_df)
#Model Creation
#Simple Recommendation model using weighted-rating
#Function to return sorted list of movies by weighted rating
#Following formula is used to calculated weighted rating Weighted Rating (WR) = (v/(v+m) * R ) + (m/(v+m)*C)

@st.cache_data
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
    #st.write(selected)
    return selected

#Function to create top movie charts for all movies and by genre
@st.cache_data
@loading_message_movie
def build_top_movie_chart(dataframe, genre=None, percentile=0.85, no_of_movies=200):
    if genre is None:
        df = dataframe
        genre=''
    else:
        df = stack_df_by_genre(dataframe)
        df = df[df['genre'] == genre]
    selected = get_top_weighted_rating(df, no_of_movies, percentile)
    st.subheader('Top ' + f'{no_of_movies} ' + f'{genre} ' + 'Movies')
    selected_movies=selected['title']
    #st.write(selected_movies)
    return selected_movies

@st.cache_data
def stack_df_by_genre(dataframe):
    metadata_temp = dataframe.copy()
    metadata_temp['genres'] = metadata_temp['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df = metadata_temp.apply(lambda x: pd.Series(
        x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    df.name = 'genre'
    df = metadata_temp.drop('genres', axis=1).join(df)
    return df

@st.cache_data
def replace_genre_json_with_list(dataframe, fieldName):
    metadata_temp = dataframe.copy()
    metadata_temp[fieldName] = metadata_temp[fieldName].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    metadata_temp[fieldName] = metadata_temp[fieldName].apply(lambda x: ','.join(map(str, x)))
    return metadata_temp    
