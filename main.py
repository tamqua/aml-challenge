from intro import greetings
import argh
from tqdm import tqdm
from matplotlib.pyplot import imshow
from sklearn.cluster import KMeans
from scipy import spatial
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.layers.core import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import data_loader
import random
import create_dataset

# displays the script title and the names of the partecipants
greetings()

def challenge():
    pass


def main():
    # we define training dataset
    training_path = os.path.join('/content/gdrive/MyDrive/dataset', 'training')

    # we define validation dataset
    validation_path = os.path.join('/content/gdrive/MyDrive/dataset', 'validation')
    gallery_path = os.path.join('/content/gdrive/MyDrive/dataset/validation', 'gallery')
    query_path = os.path.join('/content/gdrive/MyDrive/dataset/validation', 'query')
    
    #create datasets
    training_dataset = create_dataset(training_path)
    gallery_dataset = create_dataset(gallery_path)
    query_dataset = create_dataset(query_path)
    
    #show random img resized on a class folder
    plt.figure(figsize=(20,20))
    vis_folder = training_path +'/10'
    for i in range(5):
        file = random.choice(os.listdir(vis_folder))
        image_path= os.path.join(vis_folder, file)
        img=mpimg.imread(image_path)
        ax=plt.subplot(1,5,i+1)
        ax.title.set_text(file)
        plt.imshow(img)

    # we init ORB
    ORB = cv2.ORB_create()

    # we read a random image
    img_rgb = cv2.imread(gallery_paths[0])
    # convert to grayscale -> faster, easier to handle, no differences
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(img_gray,None)
    # get keypoints and descriptors
    kp, descs = orb.compute(img_gray, None)
    # draw kl
    img_orb=cv2.drawKeypoints(img_gray,kp,img_rgb,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_orb), plt.show()

if __name__ == "__main__":
    challenge()
    main()


