import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def kmc():
	data = pd.read_csv('College_Data')
	#print(data.head())
	#print(data.info())
	#print(data.describe())


	#sns.lmplot('Room.Board', 'Grad.Rate',data=data, hue='Private', fit_reg=False)
	#sns.lmplot('F.Undergrad', 'Grad.Rate',data=data, hue='Private', fit_reg=False)

	#sns.show()

	#plt.scatter(data['Grad.Rate'], data['Room.Board'], hue='Private')
	plt.show()


kmc()