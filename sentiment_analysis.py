# Sentiment Analysis (Quick Project)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'text': [
        'I love this movie, it was fantastic!',
        'What a terrible experience, I hated it.',
        'Absolutely wonderful acting and story.',
        'The film was boring and too long.',
        'Amazing direction, will watch again!',
        'Worst movie I have ever seen.'
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Convert text into vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom input
sample = ["The movie was awesome!", "I regret watching this film."]
sample_vec = vectorizer.transform(sample)
print("Predictions:", model.predict(sample_vec))
