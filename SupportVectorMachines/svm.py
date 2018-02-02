import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 


def svm():

	#---Load dataset. We are trying to predict if tumor is malignant or benign---#
	from sklearn.datasets import load_breast_cancer
	cancer = load_breast_cancer()
	#print(cancer.keys())
	#print(cancer['DESCR'])


	#---Create new dataframe with features we want, which is the data---# 
	#---column from the cancer datasets                              ---#
	features = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])


	from sklearn.model_selection import train_test_split
	x = features
	y = cancer['target']
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,
		random_state=101)

	from sklearn.svm import SVC
	supportVectorModel = SVC()
	supportVectorModel.fit(x_train, y_train)
	predictions = supportVectorModel.predict(x_test)


	from sklearn.model_selection import GridSearchCV
	param_grid = {'C':[0.1,1,10,100,1000], 'gamma': [1, 0.1, 0.001, 0.0001]}
	gridSearch = GridSearchCV(SVC(), param_grid)
	gridSearch.fit(x_train, y_train)
	print(gridSearch.best_params_)
	print(gridSearch.best_estimator_)
	print(gridSearch.best_score_)

	grid_predictions = gridSearch.predict(x_test)

	from sklearn.metrics import classification_report, confusion_matrix
	#print(confusion_matrix(y_test, predictions))
	#print(classification_report(y_test, predictions))


	print(confusion_matrix(y_test, grid_predictions))
	print(classification_report(y_test, grid_predictions))


svm()