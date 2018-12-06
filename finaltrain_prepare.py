import FunctionUnits_sk as FU
from skimage.transform import resize
from skimage import io, color, exposure
from skimage.feature import hog
import numpy as np
import time

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
    D = {}
    train = np.zeros((75750,2062))
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
                    BGR,HSV,(Hist5,Hist3),fd = BGRCues_img(img),HSVCues_img(img),HistCues_img(greyimg,area),HoG(img)
                    train[count,:] = multiappend([BGR, HSV, Hist5, Hist3, fd])
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
            np.save("train", train)
            np.save("labels", labels)

    print("valid image processed:")
    print(count)
    print("time spent (min):")
    print((time.time()-s)/60)
    np.save("train", train)
    np.save("labels", labels)
