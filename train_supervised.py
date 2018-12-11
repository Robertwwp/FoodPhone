import os
from skimage import io,color,exposure,img_as_ubyte
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.externals import joblib
import numpy as np
from PIL import Image

f = open("Supervised/food_pics/classes.txt")
folders = f.read().splitlines()
labels = np.append(np.ones(2*len(folders)), np.zeros(2*len(folders)))
traindata = np.zeros((len(labels),10))

for i, folder in enumerate(folders):

    #get the labeled Supervised Learning Model and the corresponding mask
    files = os.listdir("Supervised/food_pics/food_pics/"+folder)
    files_src = os.listdir("Supervised/food_pics/food_pics_source/"+folder)

    img1 = io.imread("Supervised/food_pics/food_pics/"+folder+"/"+files[0])
    img2 = io.imread("Supervised/food_pics/food_pics/"+folder+"/"+files[1])
    grey1, grey2 = color.rgb2grey(img1), color.rgb2grey(img2)
    cond1, cond2 = grey1!=0, grey2!=0
    img1_src = io.imread("Supervised/food_pics/food_pics_source/"+folder+"/"+files_src[0])
    img2_src = io.imread("Supervised/food_pics/food_pics_source/"+folder+"/"+files_src[1])
    if img1.shape != img1_src.shape:
        img_copy = img2_src.copy()
        img2_src = img1_src
        img1_src = img_copy

    img1_r1, img1_r2 = np.extract(cond1, img1_src), np.extract(np.invert(cond1), img1_src)
    img2_r1, img2_r2 = np.extract(cond2, img2_src), np.extract(np.invert(cond2), img2_src)
    greyimg1, greyimg2 = color.rgb2grey(img1_src), color.rgb2grey(img2_src)
    greyimg1_r1, greyimg1_r2 = np.extract(cond1, greyimg1), np.extract(np.invert(cond1), greyimg1)
    greyimg2_r1, greyimg2_r2 = np.extract(cond2, greyimg2), np.extract(np.invert(cond2), greyimg2)
    hsvimg1, hsvimg2 = color.rgb2hsv(img1_src), color.rgb2hsv(img2_src)
    hsvimg1_r1, hsvimg1_r2 = np.extract(cond1, hsvimg1), np.extract(np.invert(cond1), hsvimg1)
    hsvimg2_r1, hsvimg2_r2 = np.extract(cond2, hsvimg2), np.extract(np.invert(cond2), hsvimg2)

    #extract the rgb, hsv, histogram and position features to train the model
    traindata[i,:3], traindata[-i-1,:3] = np.mean(img1_r1,axis=0)/255, np.mean(img1_r2,axis=0)/255
    traindata[i+1,:3], traindata[-i-2,:3] = np.mean(img2_r1,axis=0)/255, np.mean(img2_r2,axis=0)/255
    traindata[i,3:6], traindata[-i-1,3:6] = np.mean(hsvimg1_r1,axis=0), np.mean(hsvimg1_r2,axis=0)
    traindata[i+1,3:6], traindata[-i-2,3:6] = np.mean(hsvimg2_r1,axis=0), np.mean(hsvimg2_r2,axis=0)
    traindata[i,6:9], traindata[-i-1,6:9] = exposure.histogram(greyimg1_r1, nbins=3)[0]/len(greyimg1_r1), exposure.histogram(greyimg1_r2, nbins=3)[0]/len(greyimg1_r2)
    traindata[i+1,6:9], traindata[-i-2,6:9] = exposure.histogram(greyimg2_r1, nbins=3)[0]/len(greyimg2_r1), exposure.histogram(greyimg2_r2, nbins=3)[0]/len(greyimg2_r2)

    energy_index1 = (np.abs(np.tile(np.arange(img1.shape[1])-img1.shape[1]/2.0, (img1.shape[0],1)))+
                     np.abs(np.tile(np.arange(img1.shape[0]).reshape(img1.shape[0],1)-img1.shape[0]/2.0, (1,img1.shape[1]))))
    energy_index2 = (np.abs(np.tile(np.arange(img2.shape[1])-img2.shape[1]/2.0, (img2.shape[0],1)))+
                     np.abs(np.tile(np.arange(img2.shape[0]).reshape(img2.shape[0],1)-img2.shape[0]/2.0, (1,img2.shape[1]))))
    traindata[i,9] = np.sum((energy_index1*cond1).ravel())/(float(np.sum(cond1)*(img1.shape[0]+img1.shape[1])/2.0))
    traindata[i+1,9] = np.sum((energy_index2*cond2).ravel())/(float(np.sum(cond2)*(img2.shape[0]+img2.shape[1])/2.0))
    traindata[-i-1,9] = np.sum((energy_index1*np.invert(cond1)))/(float(np.sum(np.invert(cond1))*(img1.shape[0]+img1.shape[1])/2.0))
    traindata[-i-2,9] = np.sum((energy_index2*np.invert(cond2)))/(float(np.sum(np.invert(cond2))*(img2.shape[0]+img2.shape[1])/2.0))
    print(traindata[i,:])
    print(traindata[i+1,:])
    print(traindata[-i-1,:])
    print(traindata[-i-2,:])

#use a SGDCClassifier
clf = linear_model.SGDClassifier()
clf_cali=CalibratedClassifierCV(clf, cv=3, method='sigmoid')
clf_cali.fit(traindata, labels)
joblib.dump(clf_cali, 'clf.pkl')
