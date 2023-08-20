import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the data from the CSV file

data =pd.read_csv('Data/Investment_news.csv' ,  low_memory=False  , encoding= 'unicode_escape')
# Create a SentimentIntensityAnalyzer object
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Create a list to store the sentiment results
sentiments = []

# Loop through the text data and classify the sentiment
for text in data['text']:
    sentiment = sia.polarity_scores(text)['compound']
    if sentiment >= 0.05:
        sentiments.append('positive')
    else:
        sentiments.append('negative')

# data['label'] = sentiments
# data.to_csv('Final_News.csv', index=False)