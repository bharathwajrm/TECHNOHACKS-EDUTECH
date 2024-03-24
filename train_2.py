import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


data = pd.read_csv("Tweets.csv")


data.dropna(subset=['text', 'airline_sentiment'], inplace=True)


vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['airline_sentiment']

model = MultinomialNB()
model.fit(X, y)

joblib.dump((model, vectorizer), 'sentiment_analysis_model.pkl')
