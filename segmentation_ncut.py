from skimage import io, color
from skimage.future import graph
import numpy as np
import os, time
import FunctionUnits_sk as FU
from skimage.segmentation import mark_boundaries,slic
from skimage.measure import regionprops
from collections import Counter

files = os.listdir("test_images")
files_existed = os.listdir("test_images_cut")
num_files, num_files_existed = len(files), len(files_existed)
#print(num_files, num_files_existed, files[num_files_existed-1], files_existed[-1])
flag = 0
start = time.time()
for file in files[num_files_existed:]:
    img = io.imread("test_images/"+file)
    try:
        superpixs = slic(img, compactness=10, n_segments=380)

        # possible to apply normalized cut before kmeans, preserve for now
        g = graph.rag_mean_color(img, superpixs, mode='similarity')
        labels = graph.cut_normalized(superpixs, g) + 1
        regions = regionprops(labels)
        for region in regions:
            if region.area < 10000:
                minr, minc, maxr, maxc = region.bbox
                outer = np.zeros((maxr - minr)*(maxc - minc) - (maxr - minr - 2)*(maxc - minc - 2))
                outer = np.append(np.append(np.append(
                                  labels[minr:maxr, minc],
                                  labels[minr:maxr, maxc-1]),
                                  labels[minr,minc+1:maxc-1].ravel()),
                                  labels[maxr-1,minc+1:maxc-1].ravel())
                outer = outer[outer != region.label]
                newlabel = Counter(outer).most_common(1)[0][0]
                labels[[x[0] for x in region.coords], [x[1] for x in region.coords]] = newlabel

        regions = regionprops(labels)

        prob = FU.Getprobsclassifier(regions, img)
        p_new = np.array(sorted([pr for pr in prob if pr!=0]))
        if (len(p_new)>1):
            #threshold = p_new[np.argmax(p_new[1:]-p_new[0:-1])+1]  #for train images, more aggressive
            threshold = (p_new.max() + p_new.mean())/2.0
        else:
            threshold=1.1

        for i, p in enumerate(prob):
            X, Y = [x[0] for x in regions[i].coords], [x[1] for x in regions[i].coords]
            if p >= threshold or p == 0:
                #labels[X,Y] = 0
                img[X,Y] = (0,0,0)
            #else:
                #labels[X,Y] = 1
        io.imsave("test_images_cut/"+file, img)
        if flag == 0:
            end = time.time()
            elasp = (end - start)*(num_files-num_files_existed)/3600.0
            print("estimate time (hr):")
            print(elasp)
            flag = 1
    except:
        io.imsave("test_images_cut/"+file, img)
        print(file)


"""out = color.label2rgb(labels, img, kind='avg')
out = mark_boundaries(out, labels, (0, 0, 0))
FU.imshow(out)
#img[[x[0] for x in regions[4].coords], [x[1] for x in regions[4].coords]] = (0,0,0)
FU.imshow(img)"""
