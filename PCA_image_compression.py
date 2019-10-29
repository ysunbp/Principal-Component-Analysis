from skimage.io import imread
from scipy import signal,ndimage
import numpy as np
import time
import scipy.io as sio
from matplotlib.pyplot import imshow,show,figure
import skimage.transform as tf
import IPython
import scipy
from sys import getsizeof
def make_grid(image_list,rows=5):
    cols = len(image_list)//rows
    row_image_list = []
    for c in range(cols):
        col_image_list = []
        for r in range(rows):
            col_image_list.append(image_list[c*rows+r])
        col_image = np.concatenate(col_image_list,axis=0)
        row_image_list.append(col_image)
    return np.concatenate(row_image_list,axis=1)
    
train_person_ids = np.arange(1,21)
train_expression_ids = np.arange(1,6)
train_image_list = []
for pid in train_person_ids:
    for eid in train_expression_ids:
        image = imread('dataset/s%d/%d.pgm'%(pid,eid))/255.
        image = tf.rescale(image,0.6)
        h,w = image.shape
        train_image_list.append(image)

train_image_collage = make_grid(train_image_list,rows=5)
#imshow(train_image_collage,cmap='gray')


test_person_ids = np.arange(21,38)
test_expression_ids = np.arange(6,11)
test_image_list = []
for pid in test_person_ids:
    for eid in test_expression_ids:
        image = imread('dataset/s%d/%d.pgm'%(pid,eid))/255.
        image = tf.rescale(image,0.6)
        h,w = image.shape
        test_image_list.append(image)
        
test_image_collage = make_grid(test_image_list,rows=5)
#imshow(test_image_collage,cmap='gray')


#PART A: compression
image_vectors = [image.flatten() for image in train_image_list]
X = np.stack(image_vectors,axis=1)
mean = np.mean(X, axis = 1)# Compute the mean.
covariance = np.cov(X)# Compute the covariance


k = 10
eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(covariance, k)# Compute the eigenfaces.
eigenvectors = np.real(eigenvectors)
for i in range(eigenvectors.shape[1]):
    imshow(eigenvectors[:,i].reshape(h,w),cmap='gray')    
    IPython.display.clear_output(True)
    show()
    
error_images = []
for image in test_image_list[:]:
    norm_image = image.flatten() - mean
    compressed_image = np.empty([k,1])
    decompressed_image = mean
    for i in range(k):
        compressed_image[i] = (np.dot(eigenvectors[:,i],norm_image.flatten()))# Compress the image into a vector of lenght k.
        decompressed_image = decompressed_image + (compressed_image[i]*eigenvectors[:,i]) # Decompress the image.
    
    error_image = np.abs(decompressed_image - image.flatten())
    error_images.append(error_image)

    imshow(np.concatenate([image.reshape(h,w),\
                          decompressed_image.reshape(h,w),\
                          error_image.reshape(h,w)],axis=-1),cmap='gray')

    IPython.display.clear_output(True)
    show()
    
print('Size before compression: %d Bytes'%image.nbytes)
print('Size after compression: %d Bytes'%compressed_image.nbytes)
           
error_images = np.stack(error_images,0)
ssd_error = (error_images**2).sum(-1).mean()
print(ssd_error)
