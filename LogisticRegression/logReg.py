import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import cufflinks as cf

def LogisticReg():

	titTrain = pd.read_csv("titanic_train.csv")

	titTrain.head()

	#Line below is used to show which data we are missing
	#sns.heatmap(titTrain.isnull(), yticklabels=False, cbar=False,cmap='viridis')

	
	sns.set_style('whitegrid')
	#sns.countplot(x='Survived', hue='Sex', data=titTrain) # Categorize by sex
	#sns.countplot(x='Survived', hue='Pclass', data=titTrain) # Categorize by passenger class
	#sns.distplot(titTrain['Age'].dropna(),kde=False,bins=30)

	titTrain['Age'].hist(bins=30,color='darkred',alpha=0.7)
	titTrain['Fare'].hist(bins=40,figsize=(8,4))

	

	cf.go_offline()
	#titTrain['Fare'].iplot(kind='hist',bins=30)


	# Cleaning the data

	plt.figure(figsize=(12,7))
	#sns.boxplot(x='Pclass', y='Age', data=titTrain)

	titTrain['Age'] = titTrain[['Age','Pclass']].apply(impute_age,axis=1)

	#sns.heatmap(titTrain.isnull(), yticklabels=False, cbar=False)
	titTrain.drop('Cabin', axis=1, inplace=True)  #Lots of cabin values null. Drop var

	titTrain.dropna(inplace=True)

	sns.heatmap(titTrain.isnull(), yticklabels=False, cbar=False)


	plt.show()



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


