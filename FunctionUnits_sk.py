import numpy as np
from skimage import io,color,exposure,img_as_ubyte,img_as_float
from matplotlib import pyplot as plt
from skimage.segmentation import slic, clear_border
from skimage.measure import regionprops
from sklearn.cluster import KMeans
from skimage.feature import hog, greycomatrix, greycoprops, daisy
from sklearn.externals import joblib

np.set_printoptions(threshold=np.inf)
clf_cali = joblib.load('clf.pkl')

def imread(path):
    img = io.imread(path)
    return img

#show image until esc
def imshow(img):
    io.imshow(img)
    plt.show()

#####################################################################
#get superpixel regionss
def getsuperpixs(img, comp, n_seg):
    sliclabels = slic(img, compactness=comp, n_segments=n_seg)
    return regionprops(sliclabels+1), sliclabels

def pre_imgs(img):
    img_norm = img/255.0
    greyimg = color.rgb2grey(img)
    hsvimg = color.rgb2hsv(img)

    return np.array(img.shape[:2], dtype=float), img_norm, greyimg, hsvimg

#3 cues for bgr color based on superpixels
def BGRCues(img_norm, regions):
    BGR = np.zeros((len(regions), 3))
    for i in range(len(regions)):
        BGR[i,:] = np.mean(img_norm[regions[i].coords[:,0],regions[i].coords[:,1]], axis=0)

    return BGR

#3 cues for hsv color based on regions
def HSVCues(hsvimg, regions):
    HSV = np.zeros((len(regions), 3))
    for i in range(len(regions)):
        HSV[i,:] = np.mean(hsvimg[regions[i].coords[:,0],regions[i].coords[:,1]], axis=0)

    return HSV

#5 cues + 3 cues
def HistCues(greyimg, regions):
    num_suppixs = len(regions)
    hist5 = np.zeros((num_suppixs,5))
    hist3 = np.zeros((num_suppixs,3))
    for i in range(num_suppixs):
        seg_img = img_as_float(greyimg[[regions[i].coords[:,0],regions[i].coords[:,1]]])
        hist5[i][:] = exposure.histogram(seg_img, nbins=5)[0]/regions[i].area
        hist3[i][:] = exposure.histogram(seg_img, nbins=3)[0]/regions[i].area

    return hist5, hist3

def HoG(img):
    fd = hog(img, orientations=8, pixels_per_cell=(32, 32),
             cells_per_block=(1, 1), feature_vector=True, multichannel=True)

    return fd

def descs(greyimg):
    descs = daisy(greyimg, step=180, radius=58, rings=2, histograms=6,
                         orientations=8)
    descs_num = descs.shape[0] * descs.shape[1]

    return np.mean(descs.reshape(descs_num, descs.shape[2]), axis=0)

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

    return np.mean(glcm,axis=0)

def multiappend(seq_features, ax):
    result = seq_features[0]
    for feature in seq_features[1:]:
        result = np.append(result, feature, axis=ax)

    return result

def Getpartcues(regions, img):
    shape, img_norm, greyimg, hsvimg = pre_imgs(img)
    BGR, HSV, Pos = BGRCues(img_norm, regions), HSVCues(hsvimg, regions), PosCues(regions, shape)
    Hist5 = HistCues(greyimg, regions)[0]
    glcm = GLCM(regions, greyimg)

    return multiappend([BGR, HSV, Hist5, Pos, glcm],1)

def Getprobsdirect(regions, greyimg):
    posprob = np.zeros(len(regions))
    center = np.array([int(greyimg.shape[0]/2), int(greyimg.shape[1]/2)])
    #var1, var2 = (np.arange(shape[0])/shape[0]).var(), (np.arange(shape[1])/shape[1]).var()
    for i, region in enumerate(regions):
        if region.area > 8000:
            posprob[i] = np.min(np.linalg.norm(region.coords-region.centroid,axis=1))**2+np.linalg.norm(region.centroid-center)**2
    posprob = posprob/np.sum(posprob)

    return posprob

def Getprobsclassifier(regions,img):
    prob = np.zeros(len(regions))
    center = np.array([int(img.shape[0]/2), int(img.shape[1]/2)])
    for i, region in enumerate(regions):
        if region.area > 5000:
            X, Y = [x[0] for x in region.coords], [x[1] for x in region.coords]
            feature = np.zeros(10)
            feature[:3] = np.mean(img[X,Y], axis=0)/255
            hsvimg = color.rgb2hsv(img)
            feature[3:6] = np.mean(hsvimg[X,Y], axis=0)
            greyimg = color.rgb2grey(img)
            feature[6:9] = exposure.histogram(greyimg[X,Y], nbins=3)[0]/region.area
            feature[9] = np.sum(np.abs(region.coords-center).ravel())/(float((region.area)*(img.shape[0]+img.shape[1])/2.0))
            prob[i] = clf_cali.predict_proba(feature.reshape(-1,10))[0][0]
            #print(feature)
            if prob[i]==0:
                prob[i] = 1e-10

    return prob/np.sum(prob)

if __name__ == '__main__':

    img = io.imread("1.jpg")
    features = Getallcues(img)
    print(features.shape)
