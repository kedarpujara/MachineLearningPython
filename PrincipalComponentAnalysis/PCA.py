import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 


def PCA():
	cancerData = load_breast_cancer()
	df = pd.DataFrame(cancerData['data'], columns=cancerData['feature_names'])
	
	print(cancerData.keys())
	print(df.head())

	scaler = StandardScaler()
	scaler.fit(df)
	scaled_data = scaler.transform(df)
	
	from sklearn.decomposition import PCA 
	pca = PCA(n_components=2)
	pca.fit(scaled_data)

	x_pca = pca.transform(scaled_data)
	print(scaled_data.shape)
	print(x_pca.shape)


	plt.figure(figsize=(10,6))
	plt.scatter(x_pca[:,0], x_pca[:,1], c=cancerData['target'], cmap='plasma')
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')


	components = pd.DataFrame(pca.components_, columns=cancerData['feature_names'])
	plt.figure(figsize=(10,6))
	sns.heatmap(components)

	plt.show()

	


PCA()