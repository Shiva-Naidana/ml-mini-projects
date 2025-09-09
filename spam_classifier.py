# Spam Email Classifier (Quick Project)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset (you can expand later)
data = {
    'text': [
        'Win money now!!!',
        'Hello, how are you?',
        'Congratulations, you have won a prize!',
        'Let’s meet tomorrow at college',
        'Claim your free coupon now',
        'Are you coming to the party?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predictions:", list(zip(X_test, y_pred)))
