import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 


def LogRegProject():


	#---Get Datafile and analyze for variables. Drop categorical variables---#
	ads = pd.read_csv('advertising.csv')

	ads.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)
	print(ads.head(2))

	#sns.distplot(ads['Age'], kde=False, bins=30)
	#sns.jointplot(x=ads['Area Income'], y=ads['Age'])



	#---Train the model using a logistic regression model---#
	from sklearn.model_selection import train_test_split
	X = ads.drop('Clicked on Ad', axis=1)
	y = ads['Clicked on Ad']

	x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
		random_state=101)

	from sklearn.linear_model import LogisticRegression
	logReg = LogisticRegression()

	logReg.fit(x_train, y_train)

	predictions = logReg.predict(x_test)



	#---Analyze how good of a metric our variable of choice is---#
	from sklearn.metrics import classification_report
	print(classification_report(y_test, predictions))

	#plt.show()





LogRegProject()