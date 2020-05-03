import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def visualize_data(train_images, num = 5, mask = None):
    '''Takes a pandas data frame of images and keys and visualizes a matrix of images'''

    if num > 10:
        raise ValueError("Trying to see too many images at once. Decrease num to be less than or equal to 10")

    #Extract images of interest
    if mask is None:
        mask = np.random.choice(train_images.index, size=(num*num), replace=False)
    images = train_images.loc[mask]['Image']
    keypoints = train_images.loc[mask][:30]

    fig, axes = plt.subplots(figsize=(num * 4, num * 4), nrows=num, ncols=num)

    for img, kp, ax in zip(images, np.array(keypoints), axes.ravel()):
        ax.imshow(np.array(img.split(' ')).reshape(96,96).astype('uint8'), cmap='gray')
        kp_pair = np.array(kp[:30]).reshape(15,2)
        ax.plot(kp_pair[:,0], kp_pair[:,1], 'ro')
        ax.axis('off')

    plt.show()

def plot_keypoint_matrix(data):
    '''Generates a scatter plot of the keypoint positions given a data set'''

    points  = data.columns # 30 points, 15 pairs

    fig, axes = plt.subplots(figsize=(20, 20), nrows=3, ncols=5)
    for kp, ax, idx in zip(np.array(points[:30]).reshape(15,2), axes.ravel(), range(15)):
        data[kp].plot( x= kp[0], y=kp[1],subplots = True, kind='scatter',ax = ax, sharex= False, sharey=False)
        ax.set_aspect('equal')
        ax.set_xlim(0,96)
        ax.set_ylim(0,96)
        ax.invert_yaxis()
        des = data[kp].describe()
        ax.scatter(x= des.loc['mean'][0],y=des.loc['mean'][1], c='y')
        ellipse_3x = Ellipse((des.loc['mean'][0],des.loc['mean'][1]),
                          width=3*des.loc['std'][0], height=3*des.loc['std'][1],
                          fill=False,color='r')
        ellipse_10x = Ellipse((des.loc['mean'][0],des.loc['mean'][1]),
                          width=10*des.loc['std'][0], height=10*des.loc['std'][1],
                          fill=False,color='g')
        ax.add_patch(ellipse_3x)
        ax.add_patch(ellipse_10x)
    plt.show()
