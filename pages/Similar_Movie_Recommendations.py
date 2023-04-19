import streamlit as st
st.title('🎬Similar Movie Recommendations🎬')

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
    return selected

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

#Top 10 movies by weighted rating

top_10=build_top_movie_chart(md_df, percentile=0.95, no_of_movies=10)
st.write(top_10)
top_crime_10=build_top_movie_chart(md_df, genre="Crime", percentile=0.90, no_of_movies=10)
st.write(top_crime_10)
top_drama_10=build_top_movie_chart(md_df, genre="Drama", percentile=0.90, no_of_movies=10)
st.write(top_drama_10)

# Content Based Reccomendation
# Using genres, spoken_languages, tagline, and overview from metadata dataset to 
# create Content based recommendation
# Create new column desc by concatenating 4 column contents spoken_languages, tagline,
# and overview from metadata dataset


@st.cache_data
def transform_meta():
    metadata_new = load_data(metadata)
    link_new=load_data(link)
    metadata_for_cont = metadata_new.copy()
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

    return metadata_for_cont

#metadata_for_cont = metadata_for_cont[metadata_for_cont.id.isin(link_new.tmdbId)]

#st.write(metadata_for_cont)

#Create n-gram and vectorize for each movie
@st.cache_data
def n_gram_vect():
    metadata_for_cont=transform_meta()
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                        min_df=30, stop_words='english')
    metadata_transformed = tf.fit_transform(metadata_for_cont.desc)

    #Calculate cosine simalirity between all movies by using words from desc column
    cosine_sim = linear_kernel(metadata_transformed, metadata_transformed)

    metadata_for_cont = metadata_for_cont.reset_index()
    titles = metadata_for_cont['title']
    indices = pd.Series(metadata_for_cont.index, index=metadata_for_cont['title'])

    return cosine_sim, titles, indices

#Function to sort movies by simalarity and return(default 30) similar movies of
# the movie passed as parameter


@st.cache_data
def get_recommendations(title, no_of_movies=30):
    cosine_sim=n_gram_vect()[0]
    titles=n_gram_vect()[1]
    indices=n_gram_vect()[2]
    idx = get_first_index(indices[title])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:no_of_movies+1]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


st.write("If you liked a certain movie we can reccomend similar ones!")
sim_movie= st.text_input("Enter the name of the movie", key="sim_movie")
sim_num= st.text_input("Enter the amount of movies you would like to be displayed", key="sim_num")
 

if sim_movie and sim_num:
    sim_num=int(sim_num)
    content_rec=get_recommendations(sim_movie, sim_num)
    st.write(content_rec)
else:
    pass



# apar_rec=get_recommendations('The Apartment', 10)

# st.write(apar_rec)