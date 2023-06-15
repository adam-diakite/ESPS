import numpy as np
import matplotlib.pyplot as plt
import h5py

import math
import re

from PIL import Image

def print_train_shapes(h5_path):
    with h5py.File(h5_path, 'r') as file:
        group_names = list(file.keys())

        for group_name in group_names:
            group = file[group_name]

            if 'train' in group:
                train_group = group['train']
                train_array = train_group[()]

                print(f"Group: {group_name}, Train Shape: {train_array.shape}")



def print_train_values(h5_path):
    with h5py.File(h5_path, 'r') as file:
        group_names = list(file.keys())

        for group_name in group_names:
            group = file[group_name]

            if 'train' in group:
                train_group = group['train']
                train_array = train_group[()]

                print(f"Group: {group_name}, Train Values:")
                print(train_array)
                print("------------------------")


def plot_train_images(h5_path):
    with h5py.File(h5_path, 'r') as file:
        group_names = list(file.keys())
        grouped_names = {}

        for group_name in group_names:
            prefix = group_name[:3]
            if prefix not in grouped_names:
                grouped_names[prefix] = []
            grouped_names[prefix].append(group_name)

        # Custom sorting function to sort group names numerically
        def sort_by_number(group_name):
            number_match = re.search(r'(\d+)$', group_name)
            if number_match:
                return int(number_match.group(1))
            return 0

        # Sort the group names based on the numeric part
        sorted_groups = sorted(grouped_names.items(), key=lambda x: sort_by_number(x[0]))

        for prefix, group_names in sorted_groups:
            num_images = len(group_names)
            num_cols = math.ceil(math.sqrt(num_images))
            num_rows = math.ceil(num_images / num_cols)

            if num_images > 0:
                if num_rows == 1 and num_cols == 1:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.set_title(f"Group: {prefix}")

                    for group_name in sorted(group_names, key=sort_by_number):
                        group = file[group_name]

                        if 'train' in group:
                            train_group = group['train']
                            train_array = train_group[()]

                            # Rotate the images counterclockwise
                            train_array = np.rot90(train_array)

                            ax.imshow(train_array, cmap='gray')
                            print(f"Group Name: {group_name}")  # Print the group name
                else:
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

                    for i, group_name in enumerate(sorted(group_names, key=sort_by_number)):
                        group = file[group_name]

                        if 'train' in group:
                            train_group = group['train']
                            train_array = train_group[()]

                            # Rotate the images counterclockwise
                            train_array = np.rot90(train_array)

                            row = i // num_cols
                            col = i % num_cols

                            if num_rows > 1:
                                ax = axes[row, col]
                            else:
                                ax = axes[col]

                            ax.imshow(train_array, cmap='gray')
                            ax.set_title(f"Group: {group_name}")
                            print(f"Group Name: {group_name}")  # Print the group name

                plt.tight_layout()
                plt.show()
            else:
                print(f"No images found for group: {prefix}")




def plot_train_images_resize(h5_path):
    with h5py.File(h5_path, 'r') as file:
        group_names = list(file.keys())
        grouped_names = {}

        for group_name in group_names:
            prefix = group_name[:3]
            if prefix not in grouped_names:
                grouped_names[prefix] = []
            grouped_names[prefix].append(group_name)

        # Custom sorting function to sort group names numerically
        def sort_by_number(group_name):
            number_match = re.search(r'(\d+)$', group_name)
            if number_match:
                return int(number_match.group(1))
            return 0

        # Sort the group names based on the numeric part
        sorted_groups = sorted(grouped_names.items(), key=lambda x: sort_by_number(x[0]))

        for prefix, group_names in sorted_groups:
            num_images = len(group_names)
            num_cols = math.ceil(math.sqrt(num_images))
            num_rows = math.ceil(num_images / num_cols)

            if num_images > 0:
                if num_rows == 1 and num_cols == 1:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.set_title(f"Group: {prefix}")

                    for group_name in sorted(group_names, key=sort_by_number):
                        group = file[group_name]

                        if 'train' in group:
                            train_group = group['train']
                            train_array = train_group[()]

                            # Rotate the images counterclockwise
                            train_array = np.rot90(train_array)

                            # Crop 24 pixels from top, bottom, right, and left
                            cropped_array = train_array[24:-24, 24:-24]

                            # Resize the array to original shape
                            resized_array = np.array(Image.fromarray(cropped_array).resize(train_array.shape))

                            ax.imshow(resized_array, cmap='gray')
                            print(f"Group Name: {group_name}")  # Print the group name
                else:
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

                    for i, group_name in enumerate(sorted(group_names, key=sort_by_number)):
                        group = file[group_name]

                        if 'train' in group:
                            train_group = group['train']
                            train_array = train_group[()]

                            # Rotate the images counterclockwise
                            train_array = np.rot90(train_array)

                            row = i // num_cols
                            col = i % num_cols

                            if num_rows > 1:
                                ax = axes[row, col]
                            else:
                                ax = axes[col]

                            # Crop 24 pixels from top, bottom, right, and left
                            cropped_array = train_array[24:-24, 24:-24]

                            # Resize the array to original shape
                            resized_array = np.array(Image.fromarray(cropped_array).resize(train_array.shape))

                            ax.imshow(resized_array, cmap='gray')
                            ax.set_title(f"Group: {group_name}")
                            print(f"Group Name: {group_name}")  # Print the group name

                plt.tight_layout()
                plt.show()
            else:
                print(f"No images found for group: {prefix}")





def gather_group_names(h5_path):
    with h5py.File(h5_path, 'r') as file:
        group_names = list(file.keys())
        grouped_names = {}

        for group_name in group_names:
            prefix = group_name[:5]
            if prefix not in grouped_names:
                grouped_names[prefix] = []
            grouped_names[prefix].append(group_name)

        return grouped_names


h5_file_path = '/home/adamdiakite/Documents/ESPS-main/Test/Open_Access_Data.hdf5'
# print_train_shapes(h5_file_path)
# print_train_values(h5_file_path)
#plot_train_images(h5_file_path)
plot_train_images_resize(h5_file_path)
