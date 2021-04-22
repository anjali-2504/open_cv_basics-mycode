import cv2 as cv
import numpy as np
import os

IMAGE_PATH =  "high_contrast_1.jpg"
STRETCH_FACTOR = 10


def main():
    cv.namedWindow("Image", cv.WINDOW_NORMAL)
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL)
    img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    ################ YOUR CODE HERE ####################
    img2 = histogram(img)
    cv.putText(img2, text= "frequency",org=(0,50),
            fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),
            thickness=2, lineType=cv.LINE_AA)
    ###################################################

    cv.imshow("Image", img)
    cv.imshow("Histogram",img2)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.destroyAllWindows()


def histogram(orig_img: np.ndarray = None) -> np.ndarray:
    """
        Returns the histogram plot of the image as a np.array

        Parameters:
            orig_img(np.ndarray): image  
    """
    img = orig_img.copy()
    (h, w) = img.shape
    freq = np.zeros((256,))

    for i in range(h):
        for j in range(w):
            freq[img[i, j]] += 1

    freq /= (h*w)
    freq *= 640

    shape = (10*int(np.max(freq)), 256*STRETCH_FACTOR)
    res = np.full(shape,fill_value=255 ,dtype=np.uint8)

    for i in range(256):
        res[-1 - 10*int(freq[i]):, i*STRETCH_FACTOR:(i+1)*STRETCH_FACTOR] = 127

    return res

if __name__ == "__main__":
    main()
