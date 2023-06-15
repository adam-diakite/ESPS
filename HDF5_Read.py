import h5py
import numpy as np
import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop, RandomResizedCrop, Resize

class H5Dataset(Dataset):
    def __init__(self, h5_path, train=True):
        self.h5_path = h5_path
        self.train = train

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as record:
            CIFAR100_MEANS = (0.485, 0.456, 0.406)
            CIFAR100_STDS = (0.229, 0.224, 0.225)

            transformations = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEANS, CIFAR100_STDS)
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEANS, CIFAR100_STDS)
                ])
            }

            keys = list(record.keys())
            group_name = keys[index]
            train_data = np.array(record[group_name]['train']).astype(np.float32)
            train_data = np.expand_dims(train_data, axis=-1)
            train_data = np.concatenate((train_data, train_data, train_data), axis=-1)

            if self.train:
                train_data = transformations['train'](train_data)
            else:
                train_data = transformations['val'](train_data)

            target_data = np.array(record[group_name]['target']).astype(np.long)

            #tumor_area = record[group_name]['tumor_area'][()]

            return train_data, target_data, group_name #tumor_area

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as record:
            return len(record)


# tumor_area = record[group_name]['tumor_area'][()]

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as record:
            return len(record)


    def __len__(self):
        with h5py.File(self.h5_path, 'r') as record:
            return len(record)

