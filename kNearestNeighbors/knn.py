import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 


def knn():

	data = pd.read_csv('ClassifiedData', index_col=0)


	#print(data.head(2))


	#---Scale the data before we train it---#
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	scaler.fit(data.drop('TARGET CLASS', axis=1))

	scaled_features = scaler.transform(data.drop('TARGET CLASS', axis=1))
	#print(scaled_features)

	data_feature = pd.DataFrame(scaled_features, columns=data.columns[:-1])
	#print(data_feature.head(5))


	#---Train the data---#
	from sklearn.model_selection import train_test_split
	x = data_feature
	y = data['TARGET CLASS']

	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, 
		random_state=101)


	from sklearn.neighbors import KNeighborsClassifier
	knnclass = KNeighborsClassifier(n_neighbors=1)
	knnclass.fit(x_train, y_train)
	predictions = knnclass.predict(x_test)


	#---Evaluate the predictions---#
	from sklearn.metrics import classification_report, confusion_matrix

	print(confusion_matrix(y_test, predictions))
	print(classification_report(y_test, predictions))


	#---Evaluate which k-value for nearest neighbors has the least error rate
	error_rate = []

	for i in range(1,40):
		knnclass2 = KNeighborsClassifier(n_neighbors=i)
		knnclass2.fit(x_train, y_train)
		pred_i = knnclass2.predict(x_test)
		error_rate.append(np.mean(pred_i != y_test))

	plt.figure(figsize=(10,6))
	plt.plot(range(1,40), error_rate, color='blue',linestyle='dashed',marker='o',
		markerfacecolor='red',markersize=10)
	plt.title('Error Rate vs K Value')
	plt.xlabel('K')
	plt.ylabel('Error Rate')
	plt.show()

knn()