'''
gan_recognizer.py

Digit recognizer for Kaggle competition using GAN as classifier.
'''


import os # path manipulation and OS resources
import pandas as pd # pandas dataframes
import numpy as np # numpy operations
import matplotlib.pyplot as plt # plots

import torch # Torch variables handler
import torch.nn as nn # Networks support
from torch.nn import init
import torch.nn.functional as F # functional
import torch.nn.parallel # parallel support
import torch.backends.cudnn as cudnn # Cuda support
import torch.optim as optim # Optimizer
import torch.utils.data as data # Data loaders
from torch.utils.data import Dataset # Dataset class

import torchvision.transforms as tf # data transforms
import torchvision.utils as vutils # image utils

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
        if 'label' in list(data.columns.values):
            targets = data['label'].values.tolist()
            data.drop('label', axis=1, inplace=True)
        else:
            targets = range(data.shape[0])

        # Converting the remaining data to values
        data = data.values.astype(float)

        # Saving data
        self.datafile = datafile
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # Main data
        self.classes = targets
        self.imgs = data

    def __getitem__(self, index):
        '''
        Returns image and target values for a given index.
        @param index Input index.
        @return The image and its respective target.
        '''

        # Get images
        label, img =  self.classes[index], self.imgs[index, :]

        # Reshape image
        img = img.reshape((28, 28, 1))

        # Transforming img
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # Return
        return img, label

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
        self.fc = nn.Sequential(
            # in 100 out: 1024
            nn.Linear(100, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # in 1024 out: 1568
            nn.Linear(1024, 1568),
            nn.BatchNorm1d(1568),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            # in: 128 x 14 x 14 out: 64 x 28 x 28
            nn.UpsamplingNearest2d(28),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # in: 64 x 28 x 28 out: 1 x 28 x 28
            nn.Conv2d(64, 1, 5, 1, 2),
            nn.Tanh(),
        )

        # Initial weights
        self._weights_init()

    def forward(self, z):
        '''
        Forward pass
        @param z input generator data.
        @return generated image.
        '''

        # Forward
        z = self.fc(z)

        # Reshaping and computing convolutional layers
        z = z.view(z.size(0), 32, 7, 7)
        z = self.conv(z)
        return z

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)

class Dnet(nn.Module):
    '''
    Discriminator net.
    '''

    def __init__(self):
        '''
        Initializes generator
        '''
        super(Dnet, self).__init__()

        # Network
        self.conv = nn.Sequential(
            # in: 1 x 28 x 28 out: 64 x 14 x 14
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # in: 64 x 14 x 14 out: 64 x 7 x 7
            nn.Conv2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            # in: 1568  out: 256
            nn.Linear(1568, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.ReLU(True),
            # in: 256, out: 20
            nn.Linear(256, 20),
            nn.Softmax(),
        )

        # Initial weights
        self._weights_init()

    def forward(self, x):
        '''
        Forward pass
        @param x input image data.
        @return classes log-probabilities.
        '''

        # COmput convolutional part
        x = self.conv(x)

        # FC part
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


if __name__ == '__main__':
    '''
    Main function.
    '''

    # Parameters
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    rslt_path = "../rslt/"
    batch_sz = 256

    # Setting outputs
    cur_mdl_path = rslt_path+'cgan_curr.pth.tar'
    imgs_path = rslt_path+'performance.pdf'

    # State
    state = {'cepoch': 0, 'nepoch' : 100, 'dlss': [], 'glss': []}
    min_lss = float('inf')

    # Setting transforms
    img_tf = tf.Compose([tf.ToTensor(), tf.Normalize([0.5], [0.5])])

    # Load dataset
    train_set = KaggleMNIST(train_path, img_tf)
    test_set = KaggleMNIST(test_path, img_tf)

    # Setting loaders
    train_load = data.DataLoader(train_set, batch_sz, True, None, 4)
    test_load = data.DataLoader(test_set, 1, False)

    # Models
    dmdl = Dnet()
    gmdl = Gnet()

    # Setting to parallel
    dmdl = torch.nn.DataParallel(dmdl).cuda()
    gmdl = torch.nn.DataParallel(gmdl).cuda()
    cudnn.benchmark = True # Inbuilt cudnn auto-tuner (fastest)

    # Loading models
    if os.path.isfile(cur_mdl_path):
        check = torch.load(cur_mdl_path)
        dmdl.load_state_dict(check['dmdl'])
        gmdl.load_state_dict(check['gmdl'])
        state = check['state']

    # Setting optimizers
    dopt = optim.Adam(dmdl.parameters(), lr=1e-4, betas=(0.5, 0.999))
    gopt = optim.Adam(gmdl.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Random vector
    zt = torch.from_numpy(np.arange(batch_sz).reshape((batch_sz, 1)) % 10)
    zr = torch.randn([batch_sz, 100])
    zt = zr.mul(zt.float().expand_as(zr))
    zt = torch.autograd.Variable(zt.cuda(), volatile=True)

    # For each epoch
    for epoch in range(state['cepoch'], state['nepoch']):

        # Setting initial losses
        edlss = 0
        eglss = 0

        # For each batch
        for i, (imgs, trgs) in enumerate(train_load):

            # Setting labels
            btsz = imgs.size(0)
            real = torch.autograd.Variable(trgs.cuda())

            # Fake labels
            fake = 10+trgs
            fake = torch.autograd.Variable(fake.cuda())

            # Images
            vimg = torch.autograd.Variable(imgs.cuda())

            # Generator vector
            zvec = torch.randn([btsz, 100])
            zr = trgs.view(trgs.size(0), 1).float()
            zvec = zvec.mul(zr.expand_as(zvec))
            zvec = torch.autograd.Variable(zvec.cuda())

            # Computing generator images
            gimg = gmdl(zvec).detach()

            # Discriminator error
            dlss = F.cross_entropy(dmdl(vimg), real)
            dlss += F.cross_entropy(dmdl(gimg), fake)

            # Update the discriminator
            dopt.zero_grad()
            dlss.backward()
            dopt.step()

            # New generator images
            zvec = torch.randn([btsz, 100])
            zr = trgs.view(trgs.size(0), 1).float()
            zvec = zvec.mul(zr.expand_as(zvec))
            zvec = torch.autograd.Variable(zvec.cuda())
            gimg = gmdl(zvec)

            # Update the generator error
            glss = F.cross_entropy(dmdl(gimg), real)
            gopt.zero_grad()
            glss.backward()
            gopt.step()

            # Update losses
            edlss += dlss.data.mean()/float(len(train_load))
            eglss += glss.data.mean()/float(len(train_load))

            # Print images
            if i % 100 == 0:
                vutils.save_image(imgs, rslt_path+'real.png',\
                normalize=True, nrow=16)
                out = gmdl(zt)
                vutils.save_image(out.data, rslt_path+'fake.png',\
                normalize=True, nrow=16)
                print 'E: [{0:03d}][{1:03d}/{2:03d}]\t D: {3:.4f}\t G: {4:.4f}\t'.format(
                    epoch, i, len(train_load), dlss.data.mean(),
                    glss.data.mean()
                )

        # Saving current loss
        state['cepoch'] = epoch+1
        state['dlss'].append(edlss)
        state['glss'].append(eglss)

        # Save model
        check = {'dmdl': dmdl.state_dict(), 'gmdl': gmdl.state_dict()}
        check['state'] = state
        torch.save(check, cur_mdl_path)

        # Plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.figure(figsize=(11.69,8.27))

        # Plot
        plt.plot(state['dlss'], 'ro-.', label='D')
        plt.plot(state['glss'], 'g--^', label='G')
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        # Save
        plt.savefig(imgs_path)
        plt.close('all')

        # Compute test results
        submission = {'Label' : []}
        for i, (imgs, _) in enumerate(test_load):

            # Computing labels
            vimg = torch.autograd.Variable(imgs.cuda(), volatile=True)
            lbls = np.argmax(dmdl(vimg).data[0].cpu().numpy())
            lbls = lbls if (lbls < 10) else lbls-10

            # Saving labels
            submission['Label'].append(lbls)

        # Saving
        nsamples = len(submission['Label'])
        submission = pd.DataFrame(submission, index = range(1, nsamples+1))
        submission.to_csv(rslt_path+'gansub.csv', index_label='ImageId')
