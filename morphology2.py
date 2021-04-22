

import cv2 as cv
import numpy as np

KERNEL_SHAPE = (3,3)
IMAGE_PATH =  "erosion.png"


def main():

    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Eroded", cv.WINDOW_NORMAL)
    img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    kernel = np.full(KERNEL_SHAPE, fill_value=0, dtype=np.uint8)

    img2 = erode(img, kernel)
    cv.imshow("Original", img)
    cv.imshow("Eroded", img2)
    cv.waitKey(0)


def erode(img: np.ndarray = None, kernel: np.ndarray = None) -> np.ndarray:
    """
            Erodes the image using kernel and returns the mat
    """
    # KERNEL
    h, w = img.shape[:2]
    res = np.zeros(img.shape, dtype=np.uint8)
    kh, kw = kernel.shape[:2]

    ky, kx = (kh-1) // 2, (kw-1) // 2

    ########################## YOUR CODE HERE ######################
	########################### TODO ###############################

   # for y in range(ky, h-ky):
    #    for x in range(kx, w-kx):
     #       addit = cv.add(img[y-ky:y+ky+1, x-kx:x+kx+1], kernel)
      #      res[y, x] = np.min(addit,axis = (0,1))
            # or res[y, x] = np.min(img[y-ky:y+ky+1, x-kx:x+kx+1],axis = (0,1))
    for y in range(ky, h-ky):
        for x in range(kx, w-kx):
            if(np.max(img[y-ky:y+ky+1, x-kx:x+kx+1],axis = (0,1))==255):
                res[y,x]=0
    ################################################################

    return res


def dilate(img: np.ndarray = None, kernel: np.ndarray = None) -> np.ndarray:
    """
            Dilates the image using kernel and returns the mat
    """
    # KERNEL
    h, w = img.shape[:2]
    res = np.full(img.shape, fill_value=255, dtype=np.uint8)
    kh, kw = kernel.shape[:2]

    ky, kx = (kh-1) // 2, (kw-1) // 2

    ########################## YOUR CODE HERE ######################
	########################### TODO ###############################

    for y in range(ky, h-ky):
        for x in range(kx, w-kx):
            addit = cv.add(img[y-ky:y+ky+1, x-kx:x+kx+1], kernel)
            res[y, x] = np.max(addit,axis=(0,1))

     ################################################################
    
    return res


if __name__ == '__main__':
    main()
