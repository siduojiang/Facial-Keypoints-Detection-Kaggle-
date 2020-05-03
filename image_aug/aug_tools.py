import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from itertools import cycle, islice
import numpy as np

def image_aug(images, keypoints, iterations = 1):
    '''
    This function will take a set of images and keypoints, and generate augmentations
        Iterations is the number of batches to generate for each image. For example,
        setting iterations to 5 will generate 5 augmented images for each input image
    '''

    #Get how many images are included
    batchsize = images.shape[0]

    #Set of transformations to apply
    #We will begin by applying an Affine transformation for all images that includes
    #    ability to rotate, scale and shear the images
    seq = iaa.Sequential([
                iaa.Affine(translate_px={"x": (0,5), "y": (0,5)}, 
                   rotate=(-5,5),
                   scale=(0.5, 1.5),
                   shear=(-16,16)),

                #perform 1 more augmentations selected from 3 possibilities
                iaa.SomeOf((0, 1), [ 
                iaa.GaussianBlur(sigma = 0.5), #add some gaussian blur
                iaa.Emboss(alpha = 0.2), 
                iaa.ReplaceElementwise(0.01, [0, 1]), #dropout some portions of the image
        ])
    ])

    #Generate arrays to store augmented images and keypoints
    #The shape of the array is the number of images (batchsize) times number of 
    #    augmented image per input image (iterations)
    new_images = np.zeros((batchsize * iterations, 96, 96, 1))
    new_keypoints = np.zeros((batchsize * iterations, 30))

    for j in range(iterations):
        for i, image in enumerate(images):    
            #Generate Keypoints objects
            kps = KeypointsOnImage.from_xy_array(keypoints[i].reshape(15,2), shape = (96,96))

            #Augment the image
            image_aug, point_aug = seq(image=image, keypoints=kps)

            new_images[j*batchsize + i] = np.expand_dims(image_aug, axis = 2)
            new_keypoints[j*batchsize + i] = point_aug.to_xy_array().ravel()

    return (new_images, new_keypoints)

def train_generator(images, keypoints, batch_size = 128, single_batch = False):
    '''
    This generator is a function compatible with Keras. This will endlessly cycle through images,
    yielding images of specified batchsize
    '''

    #endlessly cycle through all images
    image_iter = cycle(images)
    keypoints_iter = cycle(keypoints)

    while True:
        #For each batch of images, get the next slice of images and keypoints
        #The cycle object will automatically cycle back to the beginning
        #    when images run out
        image_batch = np.array(list(islice(image_iter, batch_size)))
        keypoints_batch = np.array(list(islice(keypoints_iter, batch_size)))

        #Augment each images 5 times; each batch contains each image augmented 5 times
        new_images, new_keypoints = image_aug(image_batch, keypoints_batch, 5)

        #Yield the original images with the augmented images
        yield(np.vstack([np.expand_dims(image_batch, axis = 3), new_images]), 
              np.vstack([keypoints_batch, new_keypoints]))

        if single_batch:
            break
