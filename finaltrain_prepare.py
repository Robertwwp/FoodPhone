import FunctionUnits_sk as FU
from skimage.transform import resize
from skimage import io, color, exposure, img_as_ubyte
from skimage.feature import hog, greycomatrix, greycoprops
import numpy as np
import time

# feature extraction of the training images
# RGB, HSV, Histogram and glcm dissimilarity plus correlation textures
def BGRCues_img(img, area):
    if np.max(img) <= 1:
        return np.sum(img,axis=(0,1))/area
    else:
        return np.sum(img,axis=(0,1))/(255*area)

def HSVCues_img(img, area):
    hsvimg = color.rgb2hsv(img)
    return np.sum(hsvimg,axis=(0,1))/area

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

    return np.mean(glcm,axis=0)

def multiappend(seq_features):
    result = seq_features[0]
    for feature in seq_features[1:]:
        result = np.append(result, feature, axis=0)

    return result


if __name__ == '__main__':

    """img = io.imread("t1.jpg")
    greyimg = color.rgb2grey(img)
    GLCM(greyimg)"""

    alltypes = open("classes.txt").read().splitlines()
    D = {}
    train = np.zeros((75750,30))
    labels = np.zeros(75750)
    count = 0
    for type in alltypes:
        D[type] = count
        count = count + 1

    count, filecount, skip = 0, 0, True
    s = time.time()
    for line in open("train.csv"):
        if skip:
            skip = False
        else:
            path, name = line[:-1].split(',')
            try:
                img = io.imread("train_images_cut/"+path)
                if img.shape != (512,512,3):
                    img = resize(img, (512,512,3))
                greyimg = color.rgb2grey(img)
                area = np.count_nonzero(greyimg)
                if area < 32768:
                    pass
                else:
                    BGR,HSV,(Hist5,Hist3),glcm = BGRCues_img(img,area),HSVCues_img(img,area),HistCues_img(greyimg,area),GLCM(greyimg)
                    train[count,:] = multiappend([BGR, HSV, Hist5, Hist3, glcm])
                    labels[count] = D[name]
                    count = count + 1
            except:
                print(path, name)
        filecount = filecount + 1
        if count % 1000 == 0 and count != 0:
            print("valid image processed:")
            print(count)
            print("image processed:")
            print(filecount)
            print("time spent (min):")
            print((time.time()-s)/60)
            np.save("train_glcm", train)
            np.save("labels_glcm", labels)

    # save the training data into files
    print("valid image processed:")
    print(count)
    print("time spent (min):")
    print((time.time()-s)/60)
    np.save("train_glcm", train)
    np.save("labels_glcm", labels)
