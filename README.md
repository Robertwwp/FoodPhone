# FoodPhone

link to processed train_images: https://drive.google.com/open?id=1U2MsYfffLnjfT9WHrmSWRVoroFLy9KqG

link to processed test_images: https://drive.google.com/open?id=1TTYkeoKU8R1P-tef4Tchb-BJ68HW1zVu

link to supervised training source_images: https://drive.google.com/file/d/1lwIqZnzcY9zy-Uc1iz6zRBDqWUF2pwup/view?usp=sharing

================file index================

<FunctionUnits_sk.py> Some useful functions in skimage
  
<segmentation_ncut.py> Cutting method to extract the region that is food from the image
 
<train_supervised.py> The code to train the classifier to better segment the image
  
<clf.pkl> Trained model for identify food regions in an image (training set is 200 labeled images with link above)
 
<finaltrain.py> The code to train the final svm classifier
  
<finaltest.py> Test the classifier on the test image
 
<train_final.npy> Numpy array storing all the samples with their features
 
<labels_final.npy> Numpy array storing the corresponding labels of the samples
 
 
  
