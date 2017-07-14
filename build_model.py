
# coding: utf-8

# First we import all the dependencies:



import os
import numpy as np
import scipy as sp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib

np.random.seed(42)



if __name__ == "__main__":

# We use 'wget' to fetch all the data
	print("Now Fetching!")
	os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
	os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
	os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
	os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
	os.system('gunzip *.gz')

	print("Splitting for Testing and Training")
	with open("train-images-idx3-ubyte", "rb") as f:
		X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
		X = X.reshape((60000, 28*28))

	with open("train-labels-idx1-ubyte", "rb") as f:
		y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
	indx = sss.split(X, y)

	for train_index, val_index in indx:
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]


	print("Fitting the dataset using 60 trees")
	clf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=42)
	clf.fit(X, y)
	print("Fitting Done!")

	# Pickling or dumping of the file is done using 'joblib'. The trained classifier is stored in an object called 'clf.pkl'
	joblib.dump(clf, 'classifier/clf.pkl')
	print('Pickling done!')

