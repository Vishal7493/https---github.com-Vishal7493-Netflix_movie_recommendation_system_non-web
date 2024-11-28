# Movie Recommendation System

# Importing libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data loading
# Update this with the correct path to your dataset
movies = pd.read_csv('dataset_used\dataset.csv')

# Data exploration
print(movies.describe())  # Correct usage of describe()
print(movies.isnull().sum())  # Check for null values

# Display the column names to verify
print(movies.columns)

# Feature selection
movies = movies[['id', 'title', 'overview', 'genre']]

# Creating the 'tags' column by concatenating 'overview' and 'genre'
movies['tags'] = movies['overview'].fillna('') + ' ' + movies['genre'].fillna('')

# Dropping 'overview' and 'genre' since they are now part of 'tags'
newdata = movies.drop(columns=['overview', 'genre'])

# Movie Recommendation System title
print("-------------------------------MOVIE RECOMMENDATION SYSTEM-------------------------------")
print("\n\n")

# Text vectorization using CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(newdata['tags'].values.astype('U')).toarray()

# Vector shape
print(f"Vector shape: {vector.shape}")

# Finding cosine similarity
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(movie_title):
    index = newdata[newdata['title'].str.lower() == movie_title.lower()].index  # Case insensitive search
    if not index.empty:
        index = index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movies = [newdata.iloc[i[0]].title for i in distances[1:6]]  # Exclude the input movie
        return recommended_movies
    else:
        return None

# List first N movie names for testing
def list_all_movie_names(n=20):
    return newdata['title'].head(n).tolist()

# Displaying movie list for testing
print("A total of 10,000 movie titles are available. Here are 20 of them for testing:\n")
all_movie_names = list_all_movie_names()
for name in all_movie_names:
    print(name)
print("\n")

# Getting user input
user_input = input("Which movie would you like to watch? ")

# Displaying recommendations based on user input
recommended_movies = recommend(user_input)

if recommended_movies:
    print("\nYou might also like these movies:\n")
    for movie in recommended_movies:
        print(movie.upper())
else:
    print("Sorry, no recommendations found.")
    
print("\n")
print("------------------------------------------------------------------------------------")
