import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob

def DFFTnp(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft)
    reverse_image = np.fft.ifft2(f_ishift)
    return reverse_image

def nocthFilter(fshift):
    width,height = fshift.shape
    maxpix = fshift[width//2][height//2]
    for i in range(width):
        for j in range(height):
            if i != width//2 and j != height//2:
                if abs(np.abs(fshift[i][j])-np.abs(maxpix)) < np.abs(maxpix) - 200000:
                    fshift[i][j] = 0
    return fshift

def gaussFilter(img, fshift):
    ksize=7
    kernel=np.zeros(img.shape)
    blur=cv.getGaussianKernel(ksize,-1)
    blur=np.matmul(blur,np.transpose(blur))
    kernel[0:ksize,0:ksize]=blur
    fkshift=DFFTnp(kernel)
    mult=np.multiply(fshift,fkshift)
    return mult

def main():
    path = path = os.getcwd() + '/data/'
    images = glob.glob(path + '*.png')

    for image in images:

        img = np.float32(cv.imread(image, 0))

        fshift = DFFTnp(img)
        
        #Input spectrum
        plt.subplot(231), plt.title('Input spectrum')
        plt.imshow(np.abs(fshift), 'Greys', norm=LogNorm(vmin=5))

        #Notch spectrum
        fshift_nocth = DFFTnp(img)
        fshift_nocth = nocthFilter(fshift_nocth)
        plt.subplot(232), plt.title('Notch spectrum')
        plt.imshow(np.abs(fshift_nocth), 'Greys', norm=LogNorm(vmin=5))

        #Gauss spectrum
        fshift_gauss = DFFTnp(img)
        fshift_gauss = gaussFilter(img, fshift_gauss)
        plt.subplot(233), plt.title('Gauss spectrum')
        plt.imshow(np.abs(fshift_gauss), 'Greys', norm=LogNorm(vmin=5))

        #reverse
        reverse_image_nocth = reverseDFFTnp(fshift_nocth)
        reverse_image_gauss = reverseDFFTnp(fshift_gauss)


        # output in folder
        fig = plt.gcf()
        plt.subplot(234), plt.title('Input image')
        plt.imshow(abs(img), 'Greys')
        plt.subplot(235), plt.title('Result image nocth')
        plt.imshow(abs(reverse_image_nocth), 'Greys')
        plt.subplot(236), plt.title('Result image Gauss')
        plt.imshow(abs(reverse_image_gauss), 'Greys')
        plt.show()
        fig.savefig(path[0:-6] + '/output/' + image[-9: -4] + '.jpg')

if __name__ == '__main__':
    main()