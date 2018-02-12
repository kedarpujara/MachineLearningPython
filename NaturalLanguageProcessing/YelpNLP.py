import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report



def YelpNLP():
	yelp_class = pd.read_csv('yelp.csv')
	yelp = yelp_class[(yelp_class.stars==1) | (yelp_class.stars==5)]

	stars = yelp.groupby('stars').mean()
	#print(stars)
	#print(stars.corr())

	x = yelp['text']
	y = yelp['stars']
	x_train, x_test, y_train, y_test = train_test_split(x,y,
		test_size=0.3)

	pipeline = Pipeline([
		('bow', CountVectorizer(analyzer=process_text)),
		('tfidf', TfidfTransformer()),
		('classifier', MultinomialNB())
		])
	pipeline.fit(x_train, y_train)
	predictions = pipeline.predict(x_test)

	print(classification_report(y_test, predictions))


	#print(yelp.corr())

	#print(yelp.head())


def process_text(mess):
	nopunc = [m for m in mess if m not in string.punctuation]
	nopunc = ''.join(nopunc)
	nopunc = nopunc.split()

	rm_stpwrds = [word for word in nopunc if word.lower() not in stopwords.words('english')]
	return rm_stpwrds

YelpNLP()