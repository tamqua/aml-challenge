# %%
import numpy as np
# %%

def visual_debug_orb(query_img, train_img):
    """
    Accepts 2 arguments query_img and train_img respectively the image to compare
    the image should be loaded with cv2.imread()
    """

    # check if the type of the argument passed is correct
    # ==========================================================================
    if type(query_img) != np.ndarray or type(train_img) != np.ndarray:
        raise ValueError("You must pass the image after reading it with cv2.imread()")
    # ==========================================================================



    # convert the image to grayscale
    # ==========================================================================
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
    # ==========================================================================


    # create ORB
    # ==========================================================================
    orb = cv2.ORB_create()
    # ==========================================================================


    # extract Keypoints and query descriptors
    # ==========================================================================
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
    # ==========================================================================


    # Setup the Bruteforce matcher
    # ==========================================================================
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = matcher.match(queryDescriptors,trainDescriptors)
    # ==========================================================================

    
    
    # Show the final image
    # ==========================================================================
    final_img = cv2.drawMatches(query_img_bw, queryKeypoints, train_img_bw, trainKeypoints, matches[:70],None)
    final_img = cv2.resize(final_img, (1000,650))
    cv2.imshow("Orb test better.jpg", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ==========================================================================

# %%


im = cv2.imread( os.path.join("dataset", "training", "1", "ec50k_00010008.jpg"))
im2 = cv2.imread(os.path.join("dataset", "training", "1", "ec50k_00010003.jpg"))
visual_debug_orb(im, im2)
# %%
