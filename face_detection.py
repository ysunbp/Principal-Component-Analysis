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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image = imread('FaceDetection.bmp')[:h,:]/255.
h_,w_ = image.shape

errors = []

# iterate over each part of the image
for x in range(w_-w):
    
    image_crop = image[:,x:x+w].flatten()
    
    
    norm_img = image_crop - mean
    compressed_img = np.empty([k,1])
    decompressed_img = mean
    for i in range(k):
        compressed_img[i] = (np.dot(eigenvectors[:,i], norm_img))
        decompressed_img = decompressed_img + (compressed_img[i]*eigenvectors[:,i])
        
    error = np.linalg.norm(decompressed_img - image_crop.flatten())# Your lines here. Compress and decompress the image patch and compute the reconstruction error.
    
    
    errors.append(error)
    fig, ax = plt.subplots(2,1)
    rect = patches.Rectangle((x,0),w,h,linewidth=3,edgecolor='r',facecolor='none')
    ax[0].imshow(image,cmap='gray')
    ax[0].add_patch(rect)
    ax[1].plot(np.arange(x+1)+w//2,np.array(errors))
    ax[1].set_xlim([0,w_])
    IPython.display.clear_output(True)
    show()

x_best = np.argmin(errors)# Your lines here. Find the x coordinate of the face based on the errors.
fig, ax = plt.subplots(2,1)
rect = patches.Rectangle((x_best,0),w,h,linewidth=3,edgecolor='g',facecolor='none')

ax[0].imshow(image,cmap='gray')
ax[0].add_patch(rect)
IPython.display.clear_output(True)
show()
