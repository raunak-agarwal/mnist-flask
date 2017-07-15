"""
Author: Raunak Agarwal
Title: Demo app for the second session of Sandbox 1.0, Christ University's Data Science Group.
License: MIT 
"""
# First we import all the dependencies:



import os
import numpy as np
import scipy as sp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib

np.random.seed(43)



if __name__ == "__main__":

	print("Now Fetching!")
	os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
	os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
	os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
	os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
	os.system('gunzip *.gz')

	with open("train-images-idx3-ubyte", "rb") as f:
		X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
		X = X.reshape((60000, 28*28))

	with open("train-labels-idx1-ubyte", "rb") as f:
		y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

	print("Fitting the dataset using 60 trees")
	clf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=42) #can increase the number of estimators for better accuracy.
	clf.fit(X, y)
	print("Fitting Done!")

	#The trained classifier is stored in an object called 'clf.pkl'
	joblib.dump(clf, 'classifier/clf.pkl')
	print('Pickling done!')

