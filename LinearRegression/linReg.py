import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


def LinearRegression():
	
	#---Get the Data---#
	df = pd.read_csv('USA_Housing.csv')

	df.head()
	df.describe()


	#---Graph the Data---#
	sns.pairplot(df)  #Show basic plots of all columns

	sns.distplot(df['Price']) #Show distribution plot of the Price

	df.corr()  #Show all the correlations with one another
	sns.heatmap(df.corr())  #Make a heatmap with all the correlations



	#---Create axes and variables---#
	X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
	'Avg. Area Number of Bedrooms', 'Area Population']]

	y = df['Price']  #Variable we are trying to predict




	#---Train the model---#
	from sklearn.model_selection import train_test_split

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

	from sklearn.linear_model import LinearRegression

	lm = LinearRegression()
	lm.fit(x_train, y_train)  

	#print(lm.intercept_)
	#print(lm.coef_)


	#---Create a Coefficient DataFrame---#
	# This returns a dataframe that correlates one unit increase in
	# the columns mentioned earlier for X with dollars
	cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficient'])
	

	#---Predictions from the Model---#
	predictions = lm.predict(x_test)
	plt.scatter(y_test, predictions)

	sns.distplot(y_test-predictions)



	#---Do Error Analysis---#
	from sklearn import metrics

	print("MAE:", metrics.mean_absolute_error(y_test, predictions))
	print("MSE:", metrics.mean_squared_error(y_test, predictions))
	print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))




	#---Show the graphs---#
	# Commment out this line if you don't want to show the graphs. 
	# Or comment out the lines with sns or plt if you don't want to 
	# show particular graphs
	plt.show()


LinearRegression()



