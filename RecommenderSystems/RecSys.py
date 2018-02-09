import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 

#---Creating a basic recommender system based on the MovieLens datset---#
def RecSys():
	column_names = ['user_id', 'item_id', 'rating', 'timestamp']
	df = pd.read_csv('u.data', sep='\t', names=column_names)
	#print(df.head())

	movie_titles = pd.read_csv('Movie_Id_Titles')
	#print(movie_titles.head())

	df = pd.merge(df,movie_titles,on='item_id')
	#print(df.head())


	#print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
	#print('\n')
	#print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())


	#Make a dataframe for average ratings for all the movie titles
	ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
	#print(ratings.head())

	ratings['# of Ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
	print(ratings.head())


	#ratings['# of Ratings'].hist(bins=50)
	ratings['rating'].hist(bins=50)

	plt.show()

RecSys()
