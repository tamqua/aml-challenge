
import numpy as np
import cv2
import random


#  TODO change me with the actual image
img = cv2.imread("dataset\\training\\1\\ec50k_00010002.jpg", cv2.IMREAD_COLOR)


def pick_color_channel(img, channel):

    """
    This function allows you to manually pick a color channel for the image `img`
    The choice for `channels` are (case insensitive):
    - [R]ed
    - [G]reen
    - [B]lue
    """
    # initialize the matrix which will be returned
    # --------------------------------------------------------------------------
    ret = np.zeros((img.shape[0], img.shape[1], 1), np.uint8 )

    
    # Check for the correct channel for the image to process
    # --------------------------------------------------------------------------
    if channel.lower() == "r":
        channel_index = 0
        ret = img[:,:,0]

    elif channel.lower() == "g":
        channel_index = 1
        ret = img[:,:,1]

    elif channel.lower() == "b":
        channel_index = 2
        ret = img[:,:,2]

    else:
        print(f"The value choosen was invalid")
        return

    return ret



def noise_over_image(img, prob=0.05):
    """
    Add salt and pepper noise to the image.\n
    The arguments for the function are:
    img -> the image we want to degrade
    prob -> value between 0 and 1 that denote the intensity of the degradation
    """


    # initialize the matrix which will be returned
    # --------------------------------------------------------------------------
    output = np.zeros(img.shape, np.uint8)


    # set the max treshold for white value
    # --------------------------------------------------------------------------
    treshold = 1 - prob
    

    # for each row and column of the matrix check if the condition applies to
    # change the value to white or black pixel to create a salt and pepper noise
    # --------------------------------------------------------------------------
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            random_value = random.random()
            if random_value < prob:
                output[i][j] = 0
            elif random_value > treshold:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
            return output
