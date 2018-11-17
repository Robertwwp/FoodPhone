import numpy as np
from skimage import io,color,exposure,img_as_float
from matplotlib import pyplot as plt
from skimage.segmentation import slic,mark_boundaries
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

np.set_printoptions(threshold=np.inf)

#show image until esc
def imshow(img):

    io.imshow(img)
    plt.show()

#get superpixel regions
def getsuperpixs(img):

    sliclabels = slic(img, compactness=10, n_segments=400)
    return regionprops(sliclabels)

def pre_imgs(img):

    img_norm = (img - img.mean())/img.std()
    greyimg = color.rgb2grey(img)
    greyimg_norm = (greyimg-greyimg.mean())/greyimg.std()
    hsvimg = color.rgb2hsv(img)
    hsvimg_norm = (hsvimg - img.mean())/img.std()

    return img_norm, greyimg, greyimg_norm, hsvimg, hsvimg_norm

#3 cues for bgr color based on superpixels
def BGRCues(img_norm, superpixs):

    BGR = np.zeros((len(superpixs), 3))
    for i in range(len(superpixs)):
        BGR[i] = np.mean(img_norm[superpixs[i].coords])

    return BGR

#3 cues for hsv color based on superpixs
def HSVCues(hsvimg_norm, superpixs):

    HSV = np.zeros((len(superpixs), 3))
    for i in range(len(superpixs)):
        HSV[i] = np.mean(hsvimg_norm[superpixs[i].coords])

    return HSV

#6 cues for 5 interv num_superpixal histogram with entropy + 4 cues for 3 interval histogram with entropy
def HistCues(greyimg, superpixs):

    num_suppixs = len(superpixs)
    hist5 = np.zeros((num_suppixs,6))
    hist3 = np.zeros((num_suppixs,4))

    for i in range(num_suppixs):
        seg_img = img_as_float(greyimg[superpixs[i].coords])
        hist5[i][:5] = exposure.histogram(seg_img, nbins=5)[0]/superpixs[i].area
        hist3[i][:3] = exposure.histogram(seg_img, nbins=3)[0]/superpixs[i].area
        hist5[i][:5] = (hist5[i][:5]-hist5[i][:5].mean())/hist5[i][:5].std()
        hist3[i][:3] = (hist3[i][:3]-hist3[i][:3].mean())/hist3[i][:3].std()
        hist5[i][5] = -(hist5[i][:5] @ np.log(np.abs(hist5[i][:5])/5).T)
        hist3[i][3] = -(hist3[i][:3] @ np.log(np.abs(hist3[i][:3]/3)).T)

    return hist5, hist3

#11 cues with 8 diections filter, one mean, one max, one median values
def TextureCues(greyimg_norm, superpixs):

    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            kernels.append(kernel)

    filtercues = np.zeros((len(superpixs),11))
    for k, kern in enumerate(kernels):
        fimg = np.sqrt(ndi.convolve(greyimg_norm, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(greyimg_norm, np.imag(kernel), mode='wrap')**2)
        for i in range(len(superpixs)):
            filtercues[i][k]=np.mean(fimg[superpixs[i].coords])

    filtercues[:,8]=np.mean(filtercues[:,:8], axis=1)
    filtercues[:,9]=np.amax(filtercues[:,:8], axis=1)
    filtercues[:,10]=filtercues[:,9]-np.median(filtercues[:,:8], axis=1)

    return filtercues

#4 cues for positionHSVCues(hsvimg_norm, superpixs)
def PosCues(superpixs):

    num_suppix = len(superpixs)
    PosCues = np.zeros((num_suppix,4))
    for i in range(num_suppix):
        PosCues[i,:2] = superpixs[i].centroid
        PosCues[i,2:4] = superpixs[i].local_centroid

    return PosCues

if __name__ == '__main__':

    img = io.imread("small_test/1.jpg")
    img_norm, greyimg, greyimg_norm, hsvimg, hsvimg_norm = pre_imgs(img)
    superpixs = getsuperpixs(img)
    BGRCues(img_norm, superpixs)
    HSVCues(hsvimg_norm, superpixs)
    HistCues(greyimg, superpixs)
    TextureCues(greyimg_norm, superpixs)
    PosCues(superpixs)
