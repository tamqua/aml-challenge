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


# %timeit pick_color_channel(img, "d")
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
@jit(nogil=True, parallel=True)
def noise_over_image(image, prob=0.01):
    """
    Add salt and pepper noise to the image.\n
    The arguments for the function are:
    img -> the image we want to degrade
    prob -> value between 0 and 1 that denote the intensity of the degradation
    """ 
    img = image.copy()


    # initialize the matrix which will be returned
    # --------------------------------------------------------------------------
    output = np.zeros(img.shape, np.uint8)
    
    # set the max treshold for white value
    # # --------------------------------------------------------------------------
    treshold = 1 - prob        

    # for each row and column of the matrix check if the condition applies to
    # change the value to white or black pixel to create a salt and pepper noise
    # --------------------------------------------------------------------------
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            rnd = random.random()
            if rnd < prob:
                output[i][j] = 0
            elif rnd > treshold:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]

    return output
noise_over_image(img)
# %%
def visual_debug():
    fig=plt.figure(dpi=300)
    fig.subplots_adjust(wspace = 0.5, hspace = 0.5)
    
    
    column = 4
    row = 2

    # Original
    # ==============================================================================
    img_original = img
    im1 = fig.add_subplot(row,column,1)
    im1.title.set_text("original")
    plt.imshow(img)

    noise_original = noise_over_image(img)
    fig.add_subplot(row,column,5)
    
    plt.imshow(noise_original)


    # Red 
    # ==============================================================================
    img_red = pick_color_channel(img, "r")
    fig.add_subplot(row,column,2)
    plt.imshow(img_red, cmap=plt.cm.Reds_r)

    noise_red = noise_over_image(img_red)
    fig.add_subplot(row,column,6)
    plt.imshow(noise_red, cmap=plt.cm.Reds_r)

    # Green
    # ==============================================================================
    img_green = pick_color_channel(img, "g")
    fig.add_subplot(row,column,3)
    plt.imshow(img_green, cmap=plt.cm.Greens_r)

    noise_green = noise_over_image(img_green, 0.013)
    fig.add_subplot(row,column,7)
    plt.imshow(noise_green, cmap=plt.cm.Greens_r)

    # Blue
    # ==============================================================================
    img_blue = pick_color_channel(img, "g")
    fig.add_subplot(row,column,4)
    plt.imshow(img_blue, cmap=plt.cm.Blues_r)

    noise_blue = noise_over_image(img_blue, 0.013)
    fig.add_subplot(row,column,8)
    plt.imshow(noise_blue, cmap=plt.cm.Blues_r)

    plt.show()

# # %%
visual_debug()
# %%

# %%
