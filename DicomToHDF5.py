import h5py
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import dicom2jpg
from matplotlib import pyplot as plt

dataset_dir = '/home/adamdiakite/Documents/ESPS-main/Test'
split_dir = osp.join(dataset_dir, 'LITO')
image_size = 224

#DICOM to numpy array
dicom_img_01 = "/media/adamdiakite/LaCie/PP_reformater_VS_25_04_2023/2-21-0023-2-21-0023/20180216-Thorax + IV/3-Parenchyme/000000.dcm"
dicom_dir = "/Users/user/Desktop/Patient_01"
export_location = "/Users/user/Desktop/BMP_files"


# # convert all DICOM files in dicom_dir folder to bmp, to a specified location
# dicom2jpg.dicom2bmp(dicom_dir, target_root=export_location)

#select index for each patient (largest tumor) then select the corresponding DICOM;

# convert single DICOM file to numpy.ndarray for further use
img_data = dicom2jpg.dicom2img(dicom_img_01)
print(img_data)
plt.imshow(img_data, interpolation='nearest')
plt.show()
#resized image
res = cv2.resize(img_data, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
plt.imshow(res, interpolation='nearest')
plt.show()

# #JPEG to HDF5
# save_path = './numpy.hdf5'
# img_path = '1.jpeg'
# print('image size: %d bytes'%os.path.getsize(img_path))
# hf = h5py.File(save_path, 'a') # open a hdf5 file
# img_np = np.array(Image.open(img_path))
#
# dset = hf.create_dataset('default', data=img_np)  # write the data to hdf5 file
# hf.close()  # close the hdf5 file
# print('hdf5 file size: %d bytes'%os.path.getsize(save_path))

def slice_selection(nimage):
    """Select the slice where the tumor area is the biggest
    Parameters:
    ---------------
    nimage = .nii file for which we want an image*
    """

def store_many_hdf5(h5file, images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a dataset in the file
    dataset = h5file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = h5file.create_dataset(
        "labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )

def read_many_hdf5(h5file):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images = np.array(h5file["/P0_1/train"]).astype("uint8")
    labels = np.array(h5file["/P0_1/target"]).astype("uint8")

    return images, labels

def main():
    domains = ['images']
    h5file = h5py.File(osp.join(dataset_dir, 'Open_Access_Data.hdf5'), 'w')

    for domain in domains:
        print('processing '+domain)
        #Our HDF5 where data will be contained
        h5group = h5file.create_group(domain)
        train_file_name = osp.join(split_dir,domain+'_train.txt')
        test_file = osp.join(split_dir, domain + '_test.txt')

        with open(train_file_name) as train_file:
            lines = train_file.read().splitlines()
        images = np.zeros(shape=(len(lines), image_size, image_size, 3))
        labels = np.zeros(shape=(len(lines), 1))
        for i, line in tqdm(enumerate(lines)):
            list = line.split(' ')
            image_path = osp.join(dataset_dir,list[0])
            image = Image.open(image_path)
            new_image = image.resize((image_size, image_size))
            new_image = np.array(new_image)
            label = int(list[1])
            images[i] = new_image
            labels[i]= label
        store_many_hdf5(h5group, images, labels)
    h5file.close()

if __name__ == '__main__':
    # main()
    h5file = h5py.File(osp.join(dataset_dir, 'Open_Access_Data.hdf5'), 'r')
    h5_group = h5file['/P0_1/train/']
    images,_ = read_many_hdf5(h5_group)
    img = Image.fromarray(images[0], 'RGB')
    Image._show(img)