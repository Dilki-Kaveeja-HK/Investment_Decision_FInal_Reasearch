import numpy as np
import pandas as pd
import re
import itertools
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize
# nltk.download()

from nltk.tokenize import word_tokenize
text = "Seylan Bank PLC reported earnings results for the full year ended December 31, 2021. For the full year, the company reported net interest income was LKR 23,903.91 million compared to LKR 19,810.78 million a year ago. Net income was LKR 4,653 million compared to LKR 3,038.63 million a year ago. Basic earnings per share from continuing operations was LKR 8.142 compared to LKR 5.3189 a year ago."
InputnewsforTest = word_tokenize(text)


#Reading the data
df=pd.read_csv('Data/Final_News.csv' , low_memory=False  , encoding= 'unicode_escape')
# df.head(10)

# Replace NaN values with 0
df.fillna('', inplace=True)

# checking if column have nan values

check_nan_in_df = df.isnull()
# print (check_nan_in_df) # as data dont have any NaN value, we dont need to fill them

#Getting the Labels

labels=df.label
labels.head()

# Combining important features into a single feature

df['total'] = df['title'] + ' ' + df['text']
df.head()


#PRE-PROCESSING THE DATA
stop_words = stopwords.words('english')

lemmatizer = WordNetLemmatizer()


for index, row in df.iterrows():
    filter_sentence = ''
    sentence = row['total']

    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
  #  print(sentence)
    # Tokenization
    words = nltk.word_tokenize(sentence)
    # Stopwords removal
    words = [w for w in words if not w in stop_words]
    # Lemmatization
    for words in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()

    df.loc[index, 'total'] = filter_sentence

#Displayselection = df.head()
# print(Displayselection)

# df['total'].head()

df.label = df.label.astype(str)
df.label.unique()
df.label = df.label.astype(str)
df.label = df.label.str.strip()


dict = { 'positive' : '1' , 'negative' : '0'}

df['label'] = df['label'].map(dict)

df['label'].head()
x_df = df['total']
y_df = df['label']
x_df.head()
y_df.head()




#VECOTRIZATION
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(x_df)
freq_term_matrix = count_vectorizer.transform(x_df)

tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
