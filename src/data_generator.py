"Load data to the model (Same as dataset.py but with keras)"

from typing import Callable, List, Union
from pathlib import Path
import numpy as np
import tensorflow as tf

from .npy_loader import InputOutputLoader

from .encoder_output import EncoderOutput
from .npy_loader import InputOutputLoader

class DataGenerator(tf.keras.utils.Sequence):
    """Data Generator  for keras from a list of paths to files""" 

    def __init__(self, 
                list_paths: List[Union[str, Path]], 
                list_labels: List[str],
                order_output_model: List[str],
                batch_size: int = 8,
                shuffle: bool = True,
                preprocessing: Callable = lambda x: x
                ):
        self.list_paths = list_paths  
        self.list_labels = list_labels
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.input_output_loader = InputOutputLoader(order_output_model)
        self.preprocessing = preprocessing

        # initialize first batch
        self.on_epoch_end()

    def on_epoch_end(self,):
        """Updates indexes after each epoch (starting for the epoch '0')"""
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle indexes in place

    def __len__(self):
        # Must be implemented
        """Denotes the number of batches per epoch"""
        delta = 1 if len(self.list_paths) % self.batch_size else 0 
        return len(self.list_paths) // self.batch_size + delta

    def __getitem__(self, index):
        # Must be implemented
        """To feed the model with data in training
        It generates one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of paths and labels
        list_paths_temp = [self.list_paths[k] for k in indexes]
        list_labels_temp = [self.list_labels[k] for k in indexes]

        # Generate data
        X, y = self.input_output_generation(list_paths_temp, list_labels_temp)
        return X, y
    
    def input_output_generation(self, list_path_temp: List[str], list_label_temp: List[str]): 
        """Generates and augment data containing batch_size samples
        Args:
            list_path_temp (List[str]): sublist of list_path
        Returns:
            X : numpy.array
            y : numpy.array hot-encoding
        """ 
        X_batch = []
        y_batch = []
        for path, label in zip(list_path_temp, list_label_temp): 
            npy, enc_label = self.input_output_loader(path, label)
            npy = self.preprocessing(npy)
            X_batch.append(np.expand_dims(np.expand_dims(npy,axis=0),axis=-1)) # add to list with batch dims
            y_batch.append(enc_label.index(1.0))

        return np.concatenate(X_batch, axis=0), np.array(y_batch)
