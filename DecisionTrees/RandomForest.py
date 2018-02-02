import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot 


def RandomForest():
	
	#---Grab the data and figure out what you want to test for. In  
	#---this case, it will be if kyphosis is present or absent.
	data = pd.read_csv('kyphosis.csv')

	from sklearn.model_selection import train_test_split
	x = data.drop('Kyphosis', axis=1)
	y = data['Kyphosis']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

	#---Train the data---# 
	from sklearn.ensemble import RandomForestClassifier
	rforest = RandomForestClassifier(n_estimators=100)
	rforest.fit(x_train, y_train)
	predictions = rforest.predict(x_test)

	#---Evaluate the data---#
	from sklearn.metrics import classification_report, confusion_matrix
	print(classification_report(y_test, predictions))
	print(confusion_matrix(y_test, predictions))


	#---Graph the data---#
	# features = list(data.columns[1:])
	
	# dot_data = StringIO()
	# export_graphviz(rforest, out_file=dot_data,feature_names=features,
	# 	filled=True, rounded=True)

	# graph=pydot.graph_from_dot_data(dot_data.getvalue())
	# Image(graph[0].create_png())


RandomForest()