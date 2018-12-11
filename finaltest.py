from skimage.transform import resize
from skimage import io, color, exposure, img_as_ubyte
from sklearn.externals import joblib
import numpy as np
from skimage.feature import hog, greycomatrix, greycoprops

def BGRCues_img(img):
    if np.max(img) <= 1:
        return np.sum(img,axis=(0,1))/area
    else:
        return np.sum(img,axis=(0,1))/(255*area)

def HSVCues_img(img):
    hsvimg = color.rgb2hsv(img)
    return np.mean(hsvimg,axis=(0,1))

def HistCues_img(greyimg, area):
    cond = greyimg!=0
    image = np.extract(cond, greyimg)
    hist5 = exposure.histogram(image, nbins=5)[0]/area
    hist3 = exposure.histogram(image, nbins=3)[0]/area

    return hist5, hist3

def GLCM(greyimg):
    coords = np.argwhere(greyimg != 0)
    glcm = np.zeros((9,16))
    greyimg = img_as_ubyte(greyimg)
    for i in range(1,10):
        x, y = coords[int(0.1*i*len(coords))]
        if x < 10:
            x = 10
        if y < 10:
            y = 10
        image = greyimg[x-10:x+11,y-10:y+11]
        matrix = greycomatrix(image,[1,2],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=256,normed=True)
        glcm[i-1,:8] = greycoprops(matrix, 'dissimilarity').ravel()
        glcm[i-1,8:16] = greycoprops(matrix, 'correlation').ravel()

    return np.mean(glcm,axis=0)/50

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
    #clf = joblib.load('clf_final.pkl')
    #print(clf)
    clf = joblib.load('neigh_final.pkl')
    img = io.imread("t4.jpg")
    if img.shape != (512,512,3):
        img = resize(img, (512,512,3))
    greyimg = color.rgb2grey(img)
    area = np.count_nonzero(greyimg)
    BGR,HSV,(Hist5,Hist3),glcm= BGRCues_img(img),HSVCues_img(img),HistCues_img(greyimg,area),GLCM(greyimg)
    test = multiappend([BGR, HSV, Hist5, Hist3, glcm])
    print(test)
    print(clf.predict(test.reshape(-1, len(test))))
    print(alltypes[int(clf.predict(test.reshape(-1, len(test))))])
