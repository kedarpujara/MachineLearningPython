import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import cufflinks as cf

def LogisticReg():

	titTrain = pd.read_csv("titanic_train.csv")

	titTrain.head()

	#---Graph the data to get a better understanding of relationships and correlations---#
	sns.heatmap(titTrain.isnull(), yticklabels=False, cbar=False,cmap='viridis')
	

	sns.set_style('whitegrid')
	sns.countplot(x='Survived', hue='Sex', data=titTrain) # Categorize by sex
	sns.countplot(x='Survived', hue='Pclass', data=titTrain) # Categorize by passenger class
	sns.distplot(titTrain['Age'].dropna(),kde=False,bins=30)

	titTrain['Age'].hist(bins=30,color='darkred',alpha=0.7)
	titTrain['Fare'].hist(bins=40,figsize=(8,4))

	
	cf.go_offline()
	titTrain['Fare'].iplot(kind='hist',bins=30)

	plt.figure(figsize=(12,7))
	sns.boxplot(x='Pclass', y='Age', data=titTrain)



	#---Get rid/adjust the values we are not interested in---#

	titTrain['Age'] = titTrain[['Age','Pclass']].apply(impute_age,axis=1)

	#sns.heatmap(titTrain.isnull(), yticklabels=False, cbar=False)
	titTrain.drop('Cabin', axis=1, inplace=True)  #Lots of cabin values null. Drop var

	titTrain.dropna(inplace=True)

	sexDummy = pd.get_dummies(titTrain['Sex'], drop_first=True)
	embarkDummy = pd.get_dummies(titTrain['Embarked'], drop_first=True)

	titTrain.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
	titTrain = pd.concat([titTrain, sexDummy, embarkDummy], axis=1)



	#---Train the model---#
	from sklearn.model_selection import train_test_split

	X = titTrain.drop('Survived', axis=1)
	y = titTrain['Survived']

	x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
		random_state=101)


	from sklearn.linear_model import LogisticRegression
	
	logModel = LogisticRegression()
	logModel.fit(x_train, y_train)

	predictions = logModel.predict(x_test)

	


	#---Test and evaluate the model---#
	from sklearn.metrics import classification_report

	print(classification_report(y_test, predictions))


	




	#plt.show()



#--Lots of null age values, so this replaces the null values based on 
#--average age based on Passenger class
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass==1:
			return 37
		elif Pclass==2:
			return 29
		elif Pclass==3:
			return 24
	else:
		return Age



LogisticReg()


