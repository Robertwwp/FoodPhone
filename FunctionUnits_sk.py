import numpy as np
from skimage import io,color,exposure,img_as_float
from matplotlib import pyplot as plt
from skimage.segmentation import slic,mark_boundaries
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

np.set_printoptions(threshold=np.inf)

def imread(path):
    img = io.imread(path)
    return img

#show image until esc
def imshow(img):
    io.imshow(img)
    plt.show()

#get superpixel regionss
def getsuperpixs(img):
    sliclabels = slic(img, compactness=10, n_segments=400)
    return regionprops(sliclabels)

def pre_imgs(img):
    img_norm = (img - img.mean())/255.0
    img_norm = img_norm/img_norm.std()
    greyimg = color.rgb2grey(img)
    greyimg_norm = (greyimg-greyimg.mean())/greyimg.std()
    hsvimg = color.rgb2hsv(img)
    hsvimg_norm = (hsvimg-hsvimg.mean())/hsvimg.std()

    return np.array(img.shape[:2]), img_norm, greyimg, greyimg_norm, hsvimg, hsvimg_norm

#3 cues for bgr color based on superpixels
def BGRCues(img_norm, regions):
    BGR = np.zeros((len(regions), 3))
    for i in range(len(regions)):
        BGR[i,:] = np.mean(img_norm[regions[i].coords[:,0],regions[i].coords[:,1]], axis=0)

    return BGR

#3 cues for hsv color based on regions
def HSVCues(hsvimg_norm, regions):
    HSV = np.zeros((len(regions), 3))
    for i in range(len(regions)):
        HSV[i,:] = np.mean(hsvimg_norm[regions[i].coords[:,0],regions[i].coords[:,1]], axis=0)

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
        hist5[i][:] = hist5[i][:]-hist5[i][:].mean()
        hist3[i][:] = hist3[i][:]-hist3[i][:].mean()

    return hist5, hist3

#15 cues with 6 diections x 2 frequencies filter, one mean, one max, one median values
def TextureCues(greyimg_norm, regions):
    kernels = []
    for theta in range(6):
        theta = theta / 6. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            kernels.append(kernel)

    filtercues = np.zeros((len(regions),15))
    for k, kern in enumerate(kernels):
        fimg = np.sqrt(ndi.convolve(greyimg_norm, np.real(kern), mode='wrap')**2 +
                   ndi.convolve(greyimg_norm, np.imag(kern), mode='wrap')**2)
        #print(np.max(fimg))
        for i in range(len(regions)):
            filtercues[i][k]=np.mean(fimg[[regions[i].coords[:,0],regions[i].coords[:,1]]], axis=0)

    filtercues[:,12]=np.mean(filtercues[:,:12], axis=1)
    filtercues[:,13]=np.amax(filtercues[:,:12], axis=1)
    filtercues[:,14]=filtercues[:,13]-np.median(filtercues[:,:12], axis=1)

    return filtercues*5

#10 cues for positioncues
def PosCues(regions, shape):
    num_suppix = len(regions)
    PosCues = np.zeros((num_suppix,10))
    for i in range(num_suppix):
        PosCues[i,:2] = (regions[i].centroid-0.5*shape)/shape
        PosCues[i,2:4] = (regions[i].local_centroid-0.5*shape)/shape
        PosCues[i,4:6] = (regions[i].coords[int(0.1*len(regions[i].coords))]-0.5*shape)/shape
        PosCues[i,6:8] = (regions[i].coords[int(0.5*len(regions[i].coords))]-0.5*shape)/shape
        PosCues[i,8:10] = (regions[i].coords[int(0.9*len(regions[i].coords))]-0.5*shape)/shape

    return PosCues

def multiappend(seq_features):
    result = seq_features[0]
    for feature in seq_features[1:]:
        result = np.append(result, feature, axis=1)

    return result

def Getallcues(regions, img):
    shape, img_norm, greyimg, greyimg_norm, hsvimg, hsvimg_norm = pre_imgs(img)
    BGR, HSV, (Hist5, Hist3), Texture, Pos = (BGRCues(img_norm, regionss),
    HSVCues(hsvimg_norm, regionss), HistCues(greyimg, regionss),
    TextureCues(greyimg_norm, regionss), PosCues(regionss, shape))

    return multiappend([BGR, HSV, Hist5, Hist3, Texture, Pos])

if __name__ == '__main__':

    img = io.imread("1.jpg")
    features = Getallcues(img)
    print(features.shape)
