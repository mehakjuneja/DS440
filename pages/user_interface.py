import streamlit as st
from movie_rec_app import build_top_movie_chart
from movie_rec_app import md_df

#metadata = ("https://github.com/mehakjuneja/DS440/releases/download/metadata.large/movies_metadata.csv")

user_input = st.text_input("Your username", key="user")

if user_input:
    st.write("Hello " + st.session_state.user + "!")
else:
    pass


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


