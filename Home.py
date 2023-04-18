import streamlit as st
st.title('EDA: Data Visualization of Movie Dataset')

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


md_df=load_data(metadata)
md_df['year'] = pd.to_datetime(md_df['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

md_df['id'] = md_df['id'].apply(to_int)


# load_data(rating)
# decade_graph(movies)
# genre_graph(movies)
# genre_decade_graph(movies)

links=load_data(link)
movie=load_data(movies)