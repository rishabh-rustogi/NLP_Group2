import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import nltk

train = pd.read_csv('abc.csv')
nltk.download()
clean_train_reviews = []
for i in range(0,len(train["reviews"])) :
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i],True)))

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_feature = 500000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, train("sentiment"))

clean_test_reviews = []

for i in range(0,len(test['review'])) :
    clean_test_reviews.append(" ".join(join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i],True))))
