
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


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


Accuracy = clf.score(x_test, y_test)

print(Accuracy*100)


print("---------------  2nd Factory ---------------------")

nlp.count_vectorizer.fit_transform(nlp.x_df)
input=["Seylan Bank PLC reported earnings results for the fourth quarter ended December 31, 2021. For the fourth quarter, the company reported net interest income was LKR 6,601.31 million compared to LKR 4,703.55 million a year ago. Net income was LKR 1,412.86 million compared to LKR 841.28 million a year ago. Basic earnings per share from continuing operations was LKR 2.4678 compared to LKR 1.4676 a year ago"]


freq_term_matrix2 = nlp.count_vectorizer.transform(input)


print("Result")
pred = clf.predict(freq_term_matrix2)

if (pred[0] == 1 ):
    print("Can Invest On This Company")
else:
    print("It’s better to not Invest On This Company")
