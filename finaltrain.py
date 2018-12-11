from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter

traindata = np.load("train_final.npy")
labels = np.load("labels_final.npy")

# using a linearSVC model with a squared hinge loss function
clf = LinearSVC(random_state=0, tol=1e-5, max_iter=20000)
clf.fit(traindata, labels)
joblib.dump(clf, 'clf_final.pkl')

