'''
gan_recognizer.py

Digit recognizer for Kaggle competition using GAN as classifier.
'''

import pandas as pd
import numpy as np

import torch # Torch variables handler
import torch.nn as nn # Networks support
import torch.nn.functional as F # functional
import torch.nn.parallel # parallel support
import torch.backends.cudnn as cudnn # Cuda support
import torch.optim as optim # Optimizer
import torch.utils.data as data # Data loaders
from torch.utils.data import Dataset # Dataset class

class KaggleMNIST(Dataset):
    '''
    Loads Kaggle Digit Recognizer competition MNIST dataset.
    '''
    
    def __init__(self, datafile, transform=None, target_transform=None,\
        loader=None):
        '''
        Initializes a KaggleMNIST instance.
        
        @param datafile input datafile.
        @param transform image transforms.
        @param target_transform labels transform.
        '''
        
        # Load data
        data = pd.read_csv(datafile, index_col=False)
        
        # Removing labels from data
        targets = data['label'].values.tolist()
        data.drop('label', axis=1, inplace=True)
        
        # Converting the remaining data to values
        data = data.values.astype(float)
        
        # Saving data
        self.datafile = datafile
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        # Main data
        self.classes = targets
        self.imgs = imgs
        
    def __getitem__(self, index):
        '''
        Returns image and target values for a given index.
        @param index Input index.
        @return The image and its respective target.
        '''
        
        # Get images
        label, img =  self.classes[index], self.imgs[index, :]
        
        # Reshape image
        img = img.reshape((28, 28))
        img = 
        
        # Transforming img
        if self.transform is not None:
            img = self.transform(img)

if __name__ == '__main__':