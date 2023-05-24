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

            if self.train:
                train_data1 = np.expand_dims(train_data, axis=-1)
                train_data2 = np.concatenate((train_data1, train_data1, train_data1), axis=-1)
                train_data3 = transformations['train'](train_data2)
            else:
                train_data1 = np.expand_dims(train_data, axis=-1)
                train_data2 = np.concatenate((train_data1, train_data1, train_data1), axis=-1)
                train_data3 = transformations['val'](train_data2)

            target_data = torch.randint(2, (1, 1))

            tumor_area = record[group_name]['tumor_area'][()]

            return train_data3, target_data, group_name, tumor_area

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as record:
            return len(record)


    def __len__(self):
        with h5py.File(self.h5_path, 'r') as record:
            return len(record)

