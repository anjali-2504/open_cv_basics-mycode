# import the necessary packages

import numpy as np

import cv2

import os

# Resource

# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/#:~:text=To%20detect%20colors%20in%20images,specified%20upper%20and%20lower%20range.



# ===>Global Constants-----------------------------------------------

# Woriking Directory

# wdir = "D:/Winter School(2020)/"



# Specify the image path

# image_path = wdir + "Images/colour_detect_2.jpg"



# load the image

image = cv2.imread("colour_detect_2.jpg")

#grid = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





# Windows

ImageWindow = "images"

TrackbarWindow = "TrackBars"



# define a dict of boundaries



# ---------------------------------------------------------------------





def on_trackbar(val):  # Call Back Function

    '''

        Call Back Function for All Trackbars,

        Updates the bounderies for all colors

        and refreshes the output window



        @params:val returned by trackbar which called this method,

        Note: We won't use this value,its just so that we update output only when something changes

        return: void(Nothing)

    '''



    ################## YOUR CODE HERE ########################

    ###################### TODO ##############################

    # Update All boundaries

    while(1):

        grid1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("HL", "TrackBars")

        h_max = cv2.getTrackbarPos("HH", "TrackBars")

        s_min = cv2.getTrackbarPos("SL", "TrackBars")

        s_max = cv2.getTrackbarPos("SH", "TrackBars")

        v_min = cv2.getTrackbarPos("VL", "TrackBars")

        v_max = cv2.getTrackbarPos("VH", "TrackBars")

        h1_min=cv2.getTrackbarPos("HL1", "TrackBars")

        h1_max=cv2.getTrackbarPos("HH1", "TrackBars")



    # ----------------Basically Display colours within bounds---------------------

    # Create lower and upper bound NumPy arrays from the boundariesa

        lower = np.array([h_min, s_min, v_min])

        upper = np.array([h_max, s_max, v_max])

        lower1 = np.array([h1_min, s_min, v_min])

        upper1= np.array([h1_max, s_max, v_max])       



    # find the colors within the specified boundaries and apply the mask

        mask = cv2.inRange(grid1, lower, upper)

        mask1 = cv2.inRange(grid1, lower1, upper1)

     

    #print("2")

        output = cv2.bitwise_and(image, image, mask=mask)

        output1 = cv2.bitwise_and(image, image, mask=mask1)

        output_new=cv2.add(output,output1)

        fin = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    # ------------------------------------------------------------------------------

    # cv2.imshow("img",image)



    # Refresh the image window

        img = cv2.imread("colour_detect_2.jpg")

        cv2.imshow("new", output_new)

        k = cv2.waitKey(1) & 0XFF

        if k == 27:



            break



        pass



    ################## END OF CODE ###########################



def main():

    cv2.namedWindow(TrackbarWindow)

    cv2.resizeWindow(TrackbarWindow, 600, 250)

    cv2.createTrackbar("HL", "TrackBars", 0, 180, on_trackbar)

    cv2.createTrackbar("HH", "TrackBars", 50, 180, on_trackbar)

    cv2.createTrackbar("HL1", "TrackBars", 50, 180, on_trackbar)

    cv2.createTrackbar("HH1", "TrackBars", 50, 180, on_trackbar)

    cv2.createTrackbar("SL", "TrackBars", 70, 255, on_trackbar)

    cv2.createTrackbar("SH", "TrackBars", 70, 255, on_trackbar)



    cv2.createTrackbar("VL", "TrackBars", 50, 255, on_trackbar)

    cv2.createTrackbar("VH", "TrackBars", 50, 255, on_trackbar)



    ################## YOUR CODE HERE ########################

    ###################### TODO ##############################

    # Create Trackbar window



    #

    # Trackbars for all values



    # Show window

    cv2.imshow("TrackBars", image.astype(np.uint8))

    cv2.waitKey(0)



    cv2.destroyAllWindows()





    ################## END OF CODE ###########################



if __name__ == '__main__':

    main()
