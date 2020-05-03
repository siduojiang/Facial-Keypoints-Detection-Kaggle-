import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class KaggleDataProcessor(object):
    '''Load data from Kaggle, and process into appropriate formats'''

    def __init__(self, scaler = MinMaxScaler, dir = None):
        self.dir = dir
        if dir is None:
            self.dir = os.path.dirname(os.path.realpath(__file__))

        self.scaler = scaler()

        #Import data that was downloaded from Kaggle
        self.full_training_data = pd.read_csv(self.dir + '/../data/training.zip')
        self.test_data_full = pd.read_csv(self.dir + '/../data/test.zip')

        #Import submission rows for Kaggle
        self.submission_rows = pd.read_csv(self.dir + '/../data/IdLookupTable.csv').drop('Location', axis = 1)

    def get_nonmissing_data(self):
        #Not all images have all prediction points labeled.
        #We deal with this further label. For now as a first step:
        #    drop na values from the full dataset
        full_data_nonmissing = self.full_training_data.dropna()
        train_labels_full = full_data_nonmissing.iloc[:,:30].values

        #Transform into usable np array
        train_dev_data = np.array(full_data_nonmissing.iloc[:, -1].apply(lambda x: x.split(' ')).values.tolist(), dtype=np.int)
        train_dev_data = train_dev_data.reshape(-1, 96, 96)

        return full_data_nonmissing, train_dev_data, train_labels_full

    def train_dev_test_process(self):
        '''Split the data into train and dev sets
        Fit transform data on the training set only
        Save this scaler to use for dev data'''

        _, train_dev_data, train_labels_full = self.get_nonmissing_data()

        #Data is split 80/20 for train/dev
        train_data_raw, dev_data_raw, train_labels, dev_labels = train_test_split(train_dev_data, train_labels_full, test_size=0.2, random_state = 0)

        #Another 10% of the 80 for set aside for train was split into a separate test set
        train_data_raw, test_data_raw, train_labels, test_labels = train_test_split(train_data_raw, train_labels, test_size=0.1, random_state = 0)

        num_train = train_data_raw.shape[0]
        num_dev = dev_data_raw.shape[0]
        num_test = test_data_raw.shape[0]

        train_data = self.scaler.fit_transform(train_data_raw.reshape(num_train, -1)).reshape(-1, 96, 96)
        dev_data = self.scaler.transform(dev_data_raw.reshape(num_dev, -1)).reshape(-1, 96, 96)
        test_data = self.scaler.transform(test_data_raw.reshape(num_test, -1)).reshape(-1, 96, 96)
        return train_data, dev_data, test_data, train_labels, dev_labels, test_labels

    def prepare_partial_data(self):
        '''Prepare partially labeled data for future use'''

        full_rows = self.full_training_data.isna().sum(axis = 1)
        partial_data = self.full_training_data[full_rows > 0]
        partial_labels = partial_data.iloc[:,:30]

        #Transform partially labeled image to numpy array
        partial_train = np.array(partial_data.iloc[:, -1].apply(lambda x: x.split(' ')).values.tolist(), dtype=np.int)
        partial_train_data = partial_train.reshape(-1, 96, 96)
        num_partial = partial_train_data.shape[0]
        partial_train = self.scaler.transform(partial_train_data.reshape(num_partial, -1)).reshape(-1, 96, 96)

        return partial_train, partial_labels
