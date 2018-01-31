import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


def LearningRegressionCustomer():
	customers = pd.read_csv("Ecommerce Customers")

	# customers.head()
	# customers.info()
	# customers.describe()


	# sns.set_palette("GnBu_d")
	# sns.set_style('whitegrid')
	# sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)

	#plt.show()


	X = customers[['Avg. Session Length', 'Time on App', 'Time on Website',
	'Length of Membership']]
	y = customers['Yearly Amount Spent']


	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


	from sklearn.linear_model import LinearRegression
	lm = LinearRegression()

	lm.fit(x_train, y_train)

	print(lm.intercept_)
	print("Coefficients: \t", lm.coef_)

	predictions = lm.predict(x_test)

	#plt.scatter(y_test, predictions)
	sns.distplot(y_test-predictions)
	plt.show()

LearningRegressionCustomer()