import numpy as np
import tensorflow as tf
from tensorflow.keras import applications,optimizers, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, Callback


class BestValidationModel(Callback):
    '''
    Support class for extracting the best model based on validation RMSE
        for a stream of training epochs
    '''
    __slots__ = ['val_loss', 'best_weights', 'best_epoch']

    def __init__(self):
        self.val_loss = np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        current_val_loss = logs.get('val_loss')

        #If validation loss is better than the best loss so far
        #    then update the best weights with the current epoch's weights
        if current_val_loss < self.val_loss:
            self.val_loss = current_val_loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

    def on_train_end(self, logs={}):
        print("Best epoch: ", self.best_epoch)

        #Restore the best weights
        self.model.set_weights(self.best_weights)

def model_lenet_inspired():
    '''
    A LeNet inspired CNN that gradually decreases in image size, but increases
    in channel depth. We end with Dense layers to bring flattened representation gradually down
    to 30, which is the number of target outputs

    Model uses ReLu activation functions, and the Adam optimizer
    '''
    model = Sequential()

    #The convolutional layers
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape = (96,96,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Flatten and feed to dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))

    #Output layer is a linear regression layer with 30 outputs,
    #    one for each x,y position of 15 keypoints
    model.add(Dense(30, activation=None))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


def model_dropout_before_flatten():
    '''
    A LeNet inspired CNN that gradually decreases in image size, but increases
    in channel depth. We end with Dense layers to bring flattened representation gradually down
    to 30, which is the number of target outputs

    Model uses ReLu activation functions, and the Adam optimizer
    '''
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape = (96,96,1)))
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape = (96,96,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(30, activation=None))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model
