import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

IMAGE_PATH = "high_contrast_1.jpg"
STRETCH_FACTOR = 10


def main():
    cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Histogram", cv.WINDOW_AUTOSIZE)
    img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    ################ YOUR CODE HERE ####################
    ################## TODO ############################
    img2 = histogram(img)
    pass
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
        Returns:
            res(np.ndarray): plot of Histogram as a numpy array
    """
    img = orig_img.copy()
    fig=plt.figure()
    (h, w) = img.shape
    freq = np.zeros((256,))
    hist= cv.calcHist([img], [0], None, [256], [0, 256])
    #plt.plot(hist,color='k')
    #print(hist)
    index=np.arange(256)
    #print(len(hist))
    plt.bar(index,hist[:,0])
    #plt.hist(img,bins=1)
    plt.xlim([0, 256])
    plt.ylabel("frequency")
    plt.xlabel("colour")
    canvas = FigureCanvas(fig)
    canvas.draw()
    res=np.array(fig.canvas.get_renderer()._renderer)
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    



    ################ YOUR CODE HERE ###########################
    #################### TODO #################################

    pass

    ################## END OF CODE ############################

    return res

if __name__ == "__main__":
    main()
