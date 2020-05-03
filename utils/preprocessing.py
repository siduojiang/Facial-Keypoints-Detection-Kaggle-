from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

def process_kaggle_data():

    script_dir = os.path.dirname(os.path.realpath(__file__))

    #Import data that was downloaded from Kaggle
    full_training_data = pd.read_csv(script_dir + '/../data/training.zip')
    test_data_full = pd.read_csv(script_dir + '/../data/test.zip')

    #Import submission rows for Kaggle
    submission_rows = pd.read_csv(script_dir + '/../data/IdLookupTable.csv').drop('Location', axis = 1)

    #Use MinMaxScaler to preserve sparse representations
    image_scaler = image_scaler = MinMaxScaler()

    #Not all images have all prediction points labeled.
    #We deal with this further label. For now as a first step:
    #    drop na values from the full dataset
    full_data_nonmissing = full_training_data.dropna()

    #First 30 are training labels
    names = full_data_nonmissing.iloc[:,:30].columns
    train_labels_full = full_data_nonmissing.iloc[:,:30].values

    #Transform image to numpy array
    train_dev_data = np.array(full_data_nonmissing.iloc[:, -1].apply(lambda x: x.split(' ')).values.tolist(), dtype=np.int)
    train_dev_data = train_dev_data.reshape(-1, 96, 96)

    return full_training_data, full_data_nonmissing, train_dev_data
