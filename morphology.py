import cv2 as cv
import numpy as np

KERNEL_SHAPE = (3,3) #tuple
IMAGE_PATH = "erosion.png" #image


def main():

    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Eroded", cv.WINDOW_NORMAL)
    img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    ################### YOUR CODE HERE #########################
    ###################### TODO ################################
    
    #kernel = np.full(KERNEL_SHAPE, fill_value=0, dtype=np.uint8)
    kernel = np.full(KERNEL_SHAPE, fill_value=1, dtype=np.uint8)
    #_, mask = cv.threshold(img, 220, 255, cv.THRESH_BINARY_INV)
    #img2 = None
    img2=erode(img,kernel)
    img2=dilate(img2,kernel)
    
    ############################################################
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
    print("kh = {}".format(kh))
    print("kw = {}".format(kw))
    print("ky = {}".format(ky))
    print("kw = {}".format(kx))
    #grid = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    ########################## YOUR CODE HERE ######################
    
    #blur=cv.blur(img,(5,5))
    res = cv.erode(img, kernel, iterations=1)
    ########################### TODO ###############################
    ### HINT: USE `cv.add()` ###
    #res=cv.add(np.zeros(img.shape, dtype=np.uint8),res)
    ### HINT: EROSION IS MIN OPERATION ###
    #pass

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
    print("kh = {}".format(kh))
    print("kw = {}".format(kw))
    print("ky = {}".format(ky))
    print("kw = {}".format(kx))
    #grid = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    ########################## YOUR CODE HERE ######################
    
    #blur=cv.blur(img,(5,5))
    res = cv.dilate(img, kernel, iterations=1)
    ########################### TODO ###############################
    ### HINT: USE `cv.add()` ###
    #res=cv.add(np.zeros(img.shape, dtype=np.uint8),res)
    ### HINT: EROSION IS MIN OPERATION ###
    pass

    ################################################################

    return res


if __name__ == '__main__':
    main()
