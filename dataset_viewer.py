import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop, RandomResizedCrop, Resize

path = '/home/adamdiakite/Documents/ESPS-main/Test/Open_Access_Data.hdf5'
index = 3

with h5py.File(path, 'r') as record:
    keys = list(record.keys())
    train_data = np.array(record[keys[index]]['train']).astype(np.float32)
    #target_data = np.array(record[keys[index]]['train']).astype(np.float32)

    plt.imshow(train_data)
    plt.show()

    #plt.imshow(target_data)
    plt.show()

    #print(target_data.shape)
    print(train_data.shape)
    print(len(record))


target_data = torch.rand(0,2,1, 1)
print(target_data.dtype)

