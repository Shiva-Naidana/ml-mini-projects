# Movie Recommendation System (Quick Project)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset (add more movies later if you want)
movies = {
    'title': [
        'The Matrix',
        'John Wick',
        'The Godfather',
        'The Dark Knight',
        'Pulp Fiction',
        'Interstellar',
        'Inception'
    ],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down gangsters who killed his dog.',
        'The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son.',
        'Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy.',
        'The lives of two mob hitmen, a boxer, and others intertwine in tales of violence and redemption.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.',
        'A thief who steals corporate secrets through dreams is given a chance to have his criminal history erased.'
    ]
}

df = pd.DataFrame(movies)

# Convert descriptions into vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def recommend(title, n=3):
    if title not in df['title'].values:
        return "Movie not found in dataset"
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [df['title'][i] for i, score in sim_scores[1:n+1]]
    return top_movies

# Example
print("Recommendations for 'The Matrix':", recommend('The Matrix'))
