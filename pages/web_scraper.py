import streamlit as st
st.title('IMDB Movie Poster Scraper')
from Home import md_df
import requests
from bs4 import BeautifulSoup
import urllib.request

# Define the IMDb search page URL and fetch the HTML content
search_query = st.text_input("Enter movie name to search: ", key="search")

if search_query:
    imdb_id = md_df.loc[md_df['original_title'] == search_query, 'imdb_id'].values[0]

    url = f'https://www.imdb.com/title/{imdb_id}/'
    st.write(url)
    # movie_url = f'https://www.imdb.com{result_link}'
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find the movie poster image URL
    poster_div = soup.find('div', class_='poster')
    st.write(poster_div)
    poster_url = poster_div.find('img')['src']
    

    # Download and save the image to your local directory
    poster_response = requests.get(poster_url)
    with open(f'{search_query}_poster.jpg', 'wb') as f:
        f.write(poster_response.content)

else:
    pass


#Content Based Reccomendation
#Using genres, spoken_languages, tagline, and overview from metadata dataset to 
# create Content based recommendation
#Create new column desc by concatenating 4 column contents spoken_languages, tagline,
# and overview from metadata dataset


# @st.cache_data
# def vectorization_n_gram():
#     metadata_new = load_data(metadata)
#     link_new=load_data(link)
#     metadata_for_cont = metadata_new.copy()
#     metadata_for_cont = replace_genre_json_with_list(
#         metadata_for_cont, "genres")
#     metadata_for_cont = replace_genre_json_with_list(
#         metadata_for_cont, "spoken_languages")

#     metadata_for_cont["genres"] = metadata_for_cont.genres.fillna("")
#     metadata_for_cont["spoken_languages"] = metadata_for_cont.spoken_languages.fillna(
#         "")
#     metadata_for_cont["tagline"] = metadata_for_cont.tagline.fillna("")
#     metadata_for_cont["desc"] = metadata_for_cont.genres + \
#         metadata_for_cont.spoken_languages + \
#         metadata_for_cont.overview + metadata_for_cont.tagline
#     metadata_for_cont["desc"] = metadata_for_cont.desc.fillna("")

#     metadata_for_cont = metadata_for_cont[metadata_for_cont.id.isin(link_new.tmdbId)]

#     return metadata_for_cont

# metadata_for_cont=vectorization_n_gram()

# #Create n-gram and vectorize for each movie
# tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
#                      min_df=0, stop_words='english')
# metadata_transformed = tf.fit_transform(metadata_for_cont.desc)

# #Calculate cosine simalirity between all movies by using words from desc column
# cosine_sim = linear_kernel(metadata_transformed, metadata_transformed)

# metadata_for_cont = metadata_for_cont.reset_index()
# titles = metadata_for_cont['title']
# indices = pd.Series(metadata_for_cont.index, index=metadata_for_cont['title'])

# #Function to sort movies by simalarity and return(default 30) similar movies of
# # the movie passed as parameter


# @st.cache_data
# def get_recommendations(title, no_of_movies=30):
#     idx = get_first_index(indices[title])
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:no_of_movies+1]
#     movie_indices = [i[0] for i in sim_scores]
#     return titles.iloc[movie_indices]

# apar_rec=get_recommendations('The Apartment', 10)

# print(apar_rec)