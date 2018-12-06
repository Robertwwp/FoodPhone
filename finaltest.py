from skimage.transform import resize
from skimage import io, color, exposure
from sklearn.externals import joblib
import numpy as np
from skimage.feature import hog

def BGRCues_img(img):
    return np.mean(img,axis=(0,1))/255.0

def HSVCues_img(img):
    hsvimg = color.rgb2hsv(img)
    return np.mean(hsvimg,axis=(0,1))

def HistCues_img(greyimg, area):
    hist5 = exposure.histogram(greyimg, nbins=5)[0]/area
    hist3 = exposure.histogram(greyimg, nbins=3)[0]/area

    return hist5, hist3

def HoG(img):
    fd = hog(img, orientations=8, pixels_per_cell=(32, 32),
             cells_per_block=(1, 1), feature_vector=True, multichannel=True)

    return fd

def multiappend(seq_features):
    result = seq_features[0]
    for feature in seq_features[1:]:
        result = np.append(result, feature, axis=0)

    return result


if __name__ == '__main__':
    alltypes = open("classes.txt").read().splitlines()
    clf = joblib.load('clf_final.pkl')
    img = io.imread("t2.jpg")
    if img.shape != (512,512):
        img = resize(img, (512,512))
    greyimg = color.rgb2grey(img)
    area = np.count_nonzero(greyimg)
    BGR,HSV,(Hist5,Hist3), fd = BGRCues_img(img),HSVCues_img(img),HistCues_img(greyimg,area),HoG(img)
    test = multiappend([BGR, HSV, Hist5, Hist3, fd])
    print(alltypes[int(clf.predict(test.reshape(-1, len(test))))])
