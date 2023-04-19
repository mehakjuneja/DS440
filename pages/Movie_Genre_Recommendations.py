import streamlit as st
st.title('🎭Movie Genre Recommendations🎭')

from pages.z_backend import build_top_movie_chart
from pages.z_backend import md_df


st.title('Find out the most popular movies in a certain Genre!')


add_selectbox = st.selectbox(
    'Select a genre of movies you would like to see?',
    ('Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 
     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western')
)

top_num= st.text_input("Enter the amount of movies you would like to be displayed", key="top_num")
 

if add_selectbox and top_num:
    top_num=int(top_num)
    user_top=build_top_movie_chart(md_df, genre=add_selectbox, percentile=0.90, no_of_movies=top_num)
    st.write(user_top)
else:
    pass


