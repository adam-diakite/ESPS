import os
import shutil
import nibabel as nib
import numpy as np
import glob
import pydicom
from matplotlib import pyplot as plt
import cv2

root_dir = '/media/adamdiakite/LaCie/database_paris'
single_folder_path = '/media/adamdiakite/LaCie/database/2-21-0005-2-21-0005/20110621-SCANNER THORAX/3-Parenchyme Pulmonaire'
single_dicom_path = '/media/adamdiakite/LaCie/database/2-21-0005-2-21-0005/20110621-SCANNER THORAX/2-Mediastin/000114.dcm'


lifex_script = "/home/adamdiakite/Bureau/conversion_dcm_to_nifti"

def delete_dirs_without_dcm_nii(root_dir):
    """
    Traverses the subdirectories of root_dir and deletes any subdirectory that
    does not contain both .dcm and .nii files.

    Parameters:
        root_dir (str): The root directory to start the search from.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if depth is greater than 2
        if dirpath.count(os.path.sep) - root_dir.count(os.path.sep) > 2:
            # Check if directory does not contain both .dcm and .nii files
            if not (any(filename.endswith(".dcm") for filename in filenames) and \
                    any(filename.endswith(".nii") for filename in filenames)):
                # Delete directory
                shutil.rmtree(dirpath)
                print(f"Deleted directory: {dirpath}")
            else:
                # Print directory path
                print(f"Directory contains both .dcm and .nii files: {dirpath} = Saved")


def delete_image_dirs(root_dir):
    """
    Recursively deletes all directories named 'image' under the given root directory, even if they are non-empty.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == 'single_tumour_slice' or dirname =='images':
                image_dir = os.path.join(dirpath, dirname)
                shutil.rmtree(image_dir)
                print(f'Removed directory: {image_dir}')


def display_dicom(dicom_path, dims=None):
    # Load the DICOM file
    dicom = pydicom.dcmread(dicom_path)

    # Rescale pixel values to Hounsfield units
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope
    pixel_array = dicom.pixel_array.astype(float)
    pixel_array = slope * pixel_array + intercept

    # Resize the pixel array to the desired dimensions
    if dims:
        pixel_array = cv2.resize(pixel_array, dims, interpolation=cv2.INTER_LINEAR)

    # Display the image using matplotlib
    plt.imshow(pixel_array, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


def display_dicom2(dicom_path, new_shape=None):
    dicom = pydicom.dcmread(dicom_path)
    data = dicom.pixel_array
    if new_shape is not None:
        data = cv2.resize(data, new_shape)
    plt.imshow(data, cmap=plt.cm.gray)
    plt.show()


def get_tumour_slice(folder_path):
    """
    Find largest tumor slice in nii and corresponding DCM file. and print it
    :param folder_path:
    :return: corresponding DCM file
    """

    nii_files = glob.glob(os.path.join(folder_path, '*.nii'))
    dcm_files = glob.glob(os.path.join(folder_path, '*.dcm'))

    if not nii_files:
        print('No NIfTI files found in the folder:', folder_path)
        return None

    nii_file = nii_files[0]
    img = nib.load(nii_file).get_fdata()

    # Find the index of the slice with the largest tumor area
    tumor_area = []
    for i in range(img.shape[2]):
        slice_data = img[:, :, i]
        tumor_area.append(slice_data[slice_data > 0].sum())
    if not tumor_area:
        print('No tumor found in the NIfTI file:', nii_file)
        return None

    selected_index = tumor_area.index(max(tumor_area))
    print('The index of the largest tumor area is:', selected_index)

    # Find the corresponding DCM file
    for dcm_file in dcm_files:
        if str(selected_index) in dcm_file:
            print('The corresponding DCM file is:', dcm_file)
            return dcm_file
    print('No corresponding DCM file found for the selected slice index:', selected_index)
    return None


def save_tumour_slice(folder_path):
    """
    Find the slice with the largest tumor area in a nifti file and save the corresponding DCM file
    :param folder_path: path to the folder containing the nifti file and DCM files
    :return: None
    """
    nii_files = glob.glob(os.path.join(folder_path, '*.nii'))
    dcm_files = glob.glob(os.path.join(folder_path, '*.dcm'))

    if not nii_files:
        print('No NIfTI files found in the folder:', folder_path)
        return
    if len(nii_files) > 1:
        print('Multiple NIfTI files found in the folder:', folder_path)
        return

    nii_file = nii_files[0]
    img = nib.load(nii_file).get_fdata()

    # Find the index of the slice with the largest tumor area
    tumor_areas = []
    for i in range(img.shape[2]):
        slice_data = img[:, :, i]
        tumor_areas.append(slice_data[slice_data > 0].sum())
    if not any(tumor_areas):
        print('No tumor found in the NIfTI file:', nii_file)
        return

    max_tumor_area_index = max(range(len(tumor_areas)), key=tumor_areas.__getitem__)

    # Create a folder to save the DCM file in
    output_folder = os.path.join(os.path.dirname(dcm_files[0]), 'single_tumour_slice')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the corresponding DCM file for the slice with the largest tumor area
    for dcm_file in dcm_files:
        dcm_index = int(os.path.splitext(os.path.basename(dcm_file))[0])
        if dcm_index == max_tumor_area_index + 1:
            output_file = os.path.join(output_folder, os.path.basename(dcm_file))
            shutil.copyfile(dcm_file, output_file)
            print(f"Saved largest tumor area DCM slice {dcm_index} as {output_file}")

    print(f"The DCM file for the slice with the largest tumor area has been saved in {output_folder}")



def save_all_tumour_slices(folder_path):
    """
    Find the slices with tumors in a nifti file and save the corresponding DCM files
    :param folder_path: path to the folder containing the nifti file and DCM files
    :return: None
    """
    nii_files = glob.glob(os.path.join(folder_path, '*.nii'))
    dcm_files = glob.glob(os.path.join(folder_path, '*.dcm'))

    if not nii_files:
        print('No NIfTI files found in the folder:', folder_path)
        return
    if len(nii_files) > 1:
        print('Multiple NIfTI files found in the folder:', folder_path)
        return

    nii_file = nii_files[0]
    img = nib.load(nii_file).get_fdata()

    # Find the slices with tumors
    tumor_slices = []
    for i in range(img.shape[2]):
        slice_data = img[:, :, i]
        if slice_data[slice_data > 0].sum() > 0:
            tumor_slices.append(i)

    if not tumor_slices:
        print('No tumor found in the NIfTI file:', nii_file)
        return

    # Create a folder to save the DCM files in
    output_folder = os.path.join(os.path.dirname(dcm_files[0]), 'tumour_slices')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the corresponding DCM files for the slices with tumors
    for dcm_file in dcm_files:
        dcm_index = int(os.path.splitext(os.path.basename(dcm_file))[0]) - 1
        if dcm_index in tumor_slices:
            output_file = os.path.join(output_folder, os.path.basename(dcm_file))
            shutil.copyfile(dcm_file, output_file)
            print(f"Saved tumor slice {dcm_index + 1} as {output_file}")

    print(f"The DCM files for the slices with tumors have been saved in {output_folder}")



def process_directories_general(root_directory):
    """
    Recursively process directories and apply save_tumour_slice() to each directory
    :param root_directory: the root directory to process
    """
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for dirname in dirnames:
            dir_full_path = os.path.join(dirpath, dirname)
            #Function I want to apply to the whole dataset is here.
            copy_nii_files(dir_full_path)




def copy_nii_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".nii"):
                source_path = os.path.join(root, file_name)
                destination_folder = os.path.join(root, "images")

                # Check if the "images" folder already exists
                if os.path.exists(destination_folder):
                    print("Error, Images folder already exists for this folder ")
                    continue

                # Create the "images" folder
                os.makedirs(destination_folder)

                destination_path = os.path.join(destination_folder, file_name)
                shutil.copy(source_path, destination_path)

    print("NII files copied to the 'images' subfolder.")



#delete_image_dirs((root_dir))
process_directories_general('/media/adamdiakite/LaCie/database_dijon')

# Cleaning the dataset : Deletes the folders where there aren't any segmentations
# delete_dirs_without_dcm_nii(root_dir)

# Print the selected single slice + corresponding .dcm file
# print(get_tumour_slice(single_folder))

# For a patient, saves the image with the most lesion area AND all images with lesion areas in two separate folders.
# print(save_all_tumour_slices(single_folder))
# print(save_tumour_slice(single_folder))

# Going through all databases and saving the single selected .dcm file into a separate folder.
# print(process_directories(folder_path))


