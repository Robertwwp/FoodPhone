from skimage import io, color, morphology
from skimage.future import graph
import numpy as np
import FunctionUnits_sk as FU
from skimage.segmentation import find_boundaries,mark_boundaries,slic
from skimage.measure import regionprops
from collections import Counter

img = io.imread("small_test/6.jpg")
superpixs = slic(img, compactness=10, n_segments=400)
g = graph.rag_mean_color(img, superpixs, mode='similarity')

labels = graph.cut_normalized(superpixs, g)
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
        for coord in region.coords:
            labels[coord[0], coord[1]] = newlabel

out = color.label2rgb(labels, img, kind='avg')
out = mark_boundaries(out, labels, (0, 0, 0))
FU.imshow(out)
