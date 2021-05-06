# %%
import numpy as np
import cv2
import random
from numba import prange, jit, parfors
import matplotlib.pyplot as plt

# %%
#  TODO change me with the actual image
img = cv2.imread("dataset\\training\\1\\ec50k_00010002.jpg", cv2.IMREAD_COLOR)


# %%
@jit(nogil=True)
def pick_color_channel(image, channel):

    """
    This function allows you to manually pick a color channel for the image `img`
    The choice for `channels` are (case insensitive):
    - [R]ed
    - [G]reen
    - [B]lue
    """
    # initialize the matrix which will be returned
    # --------------------------------------------------------------------------
    img = image.copy()
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    
    
    
    # Check for the correct channel for the image to process
    # --------------------------------------------------------------------------
    if channel.lower() == "r":
        return red

    elif channel.lower() == "g":
        return green

    elif channel.lower() == "b":
        return blue
    else:
        raise ValueError("The channel specified in `pick_color_channel(image, channel)` is incorrect.\nPlease use one of the following: `r` `g` `b`")

    # else:
    #     print(f"The value choosen was invalid")
    #     return wrong


%timeit pick_color_channel(img, "d")
# Visual debug 
# ==============================================================================
# plt.imshow(pick_color_channel(img, "r"), cmap=plt.cm.Reds_r)
# plt.imshow(pick_color_channel(img, "g"), cmap=plt.cm.Greens_r)
# plt.imshow(pick_color_channel(img, "b"), cmap=plt.cm.Blues_r)
# plt.imshow(pick_color_channel(img, "b"), cmap=plt.cm.gray)
# %%


# ==============================================================================
# The jit function decorator of the packet Numba allows to compile the code to 
# make it run faster. In fact the results measured with %timeit magic methods
# are the following:
# Normal -> 273 ms ± 2.36 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# Numba ->  3.59 ms ± 432 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
# ==============================================================================
@jit(nopython=True, nogil=True, parallel=True)
def noise_over_image(img, prob=0.015):
    """
    Add salt and pepper noise to the image.\n
    The arguments for the function are:
    img -> the image we want to degrade
    prob -> value between 0 and 1 that denote the intensity of the degradation
    """


    # initialize the matrix which will be returned
    # --------------------------------------------------------------------------
    output = np.zeros(img.shape, np.uint8)
    
    # initialize the random matrix to compare for checking if inserting the noise
    # --------------------------------------------------------------------------
    random_value = np.random.random(img.shape)
    

    # set the max treshold for white value
    # # --------------------------------------------------------------------------
    treshold = np.ones(img.shape) - prob
        

    # for each row and column of the matrix check if the condition applies to
    # change the value to white or black pixel to create a salt and pepper noise
    # --------------------------------------------------------------------------
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            if random_value[i][j][0] < prob:
                output[i][j] = [0,0,0]
            elif random_value[i][j][0] > treshold[i][j][0]:
                output[i][j] = [255,255,255]
            else:
                output[i][j] = img[i][j]
    return output
%timeit noise_over_image(img)
# %%
