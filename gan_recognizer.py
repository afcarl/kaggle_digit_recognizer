'''
gan_recognizer.py

Digit recognizer for Kaggle competition using GAN as classifier.
'''

import pandas as pd
import numpy as np
import argparse

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
        img = img.reshape((1, 28, 28))
        
        # Transforming img
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Return
        return img, target
    
    def __len__(self):
        '''
        Returns number of samples
        @return Number of samples.
        '''
        return len(self.classes)

class Gnet(nn.Module):
    '''
    Generator net.
    '''
    
    def __init__(self):
        '''
        Initializes generator
        '''
        super(Gnet, self).__init__()
        
        # Network
        self.layers = nn.Sequential(
            # in: 100 x 1 x 1 out: 64 x 4 x 4
            nn.Conv2d(100, 1024, 1, bias=False),
            nn.PixelShuffle(4),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            # in: 64 x 4 x 4 out: 128 x 8 x 8
            nn.Conv2d(64, 512, 1, bias=False),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            # in: 128 x 8 x 8 out: 128 x 16 x 16
            nn.Conv2d(128, 512, 1, bias=False),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            # in: 128 x 16 x 16 out: 128 x 32 x 32
            nn.Conv2d(128, 512, 1, bias=False),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            # in: 128 x 32 x32 out: 1 x 28 x 28
            nn.Conv2d(128, 128, 5, bias=False),
            nn.Conv2d(128, 1, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z):
        '''
        Forward pass
        @param z input generator data.
        @return generated image.
        '''
        return self.layers(z)
    
class Dnet(nn.Module):
    '''
    Discriminator net.
    '''
    
    def __init__(self):
        '''
        Initializes generator
        '''
        super(Gnet, self).__init__()
        
        # Network
        self.layers = nn.Sequential(
            # in: 1 x 28 x 28 out: 64 x 14 x 14
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # in: 64 x 14 x 14 out: 128 x 7 x 7
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # in: 128 x 7 x 7 out: 256 x 4 x 4
            nn.Conv2d(128, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # in: 256 x 4 x 4 out: 128 x 2 x 2
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # in: 512 x 2 x 2 out: 512 x 1 x 1
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.Conv2d(512, 10, 1, bias=False),
            nn.LogSigmoid()
        )
        
    def forward(self, x):
        '''
        Forward pass
        @param x input image data.
        @return classes log-probabilities.
        '''
        x = self.layers(x)
        return x.view(x.size(0), -1)
        

if __name__ == '__main__':
    '''
    Main function.
    '''
    
    # Parse