from sklearn.externals import joblib
from sklearn.svm import SVR
import numpy as np
from collections import Counter

#4667 useless
traindata = np.load("train.npy")[:-4667]
labels = np.load("labels.npy")[:-4667]

clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(traindata, labels)
joblib.dump(clf, 'clf_final.pkl')
