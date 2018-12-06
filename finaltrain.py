from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
from collections import Counter

#4667 useless
traindata = np.load("train.npy")[:-4667]
labels = np.load("labels.npy")[:-4667]

clf = SVC(gamma='auto')
clf.fit(traindata, labels)
joblib.dump(clf_cali, 'clf_final.pkl')
