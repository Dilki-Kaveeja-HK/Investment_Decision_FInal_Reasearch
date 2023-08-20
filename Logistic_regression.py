
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

import nlpBase as nlp


#Splitting data into train and test data

x_train, x_test, y_train, y_test = train_test_split(nlp.tf_idf_matrix,
                                   nlp.y_df, random_state=0)

y_train.head()

#LOGISTIC REGRESSION


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Accuracy = logreg.score(x_test, y_test)

print(Accuracy*100)


print("---------------  2nd Factory ---------------------")

nlp.count_vectorizer.fit_transform(nlp.x_df)
input=[" ratio of 3.7x.Average trailing P/E is 3x in the Banks industry in Sri Lanka"]
# input=["switched this simply by defining a different codec package"]

freq_term_matrix2 = nlp.count_vectorizer.transform(input)



print("Result")
pred = logreg.predict(freq_term_matrix2)

if (pred[0] == 1 ):
    print("Can Invest On This Company")
else:
    print("It’s better to not Invest On This Company")