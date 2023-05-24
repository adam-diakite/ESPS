import os
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from scipy import ndimage
import h5py
from matplotlib.widgets import Slider
import nibabel as nib
import cv2
import re

root_dir = '/home/adamdiakite/Images/Patient_Vincent_PP'
path = '/home/adamdiakite/Images/Patient_Vincent_PP/0/scans'


def read_dicom_folder(folder_path):
    # Load DICOM files from folder
    dicom_files = []
    for dirName, subdirList, fileList in os.walk(folder_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # Check whether the file is DICOM
                dicom_files.append(os.path.join(dirName, filename))

    # Read DICOM data
    slices = [pydicom.read_file(f) for f in dicom_files]
    slices.sort(key=lambda x: x.ImagePositionPatient[2])

    # Convert DICOM to NIfTI
    nifti_image = nib.Nifti1Image(np.stack([s.pixel_array for s in slices]), np.eye(4))

    # Display NIfTI volume in sagittal, axial, and coronal views
    volume_data = np.rot90(nifti_image.get_fdata(), k=1)  # Rotate to correct orientation

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axial_im = axs[0].imshow(volume_data[:, int(volume_data.shape[1] / 2), :], cmap="gray", origin="lower")
    axs[0].set_title("Axial View")
    sagittal_im = axs[1].imshow(volume_data[int(volume_data.shape[0] / 2), :, :], cmap="gray", origin="lower")
    axs[1].set_title("Sagittal View")
    coronal_im = axs[2].imshow(volume_data[:, :, int(volume_data.shape[2] / 2)], cmap="gray", origin="lower")
    axs[2].set_title("Coronal View")

    # Add slider to control slice selection
    slice_axis = 0  # Starting axis
    slice_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Create slider axis
    slice_slider = Slider(slice_slider_ax, 'Slice', 0, volume_data.shape[slice_axis] - 1,
                          valinit=int(volume_data.shape[slice_axis] / 2), valstep=1)  # Create slider

    # Define function to update the slices based on slider value
    # Define function to update the slices based on slider value
    def update(val):
        slice_index = int(slice_slider.val)
        if slice_axis == 0:
            sagittal_im.set_data(volume_data[slice_index, :, :])
            axial_im.set_data(volume_data[:, slice_index, :])
            coronal_im.set_data(volume_data[:, :, slice_index])
        elif slice_axis == 1:
            sagittal_im.set_data(volume_data[:, slice_index, :])
            axial_im.set_data(volume_data[slice_index, :, :])
            coronal_im.set_data(volume_data[:, :, slice_index])
        elif slice_axis == 2:
            sagittal_im.set_data(volume_data[:, :, slice_index])
            axial_im.set_data(volume_data[:, slice_index, :])
            coronal_im.set_data(volume_data[slice_index, :, :])
        fig.canvas.draw_idle()

    # Attach update function to slider
    slice_slider.on_changed(update)

    plt.show()


def read_dicom_folder_mask(folder_path):
    # Load DICOM files from folder
    dicom_files = []
    for dirName, subdirList, fileList in os.walk(folder_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # Check whether the file is DICOM
                dicom_files.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():  # Check whether the file is NIfTI
                mask_path = os.path.join(dirName, filename)

    # Read DICOM data
    slices = [pydicom.read_file(f) for f in dicom_files]
    slices.sort(key=lambda x: x.ImagePositionPatient[2])

    # Convert DICOM to NIfTI
    nifti_image = nib.Nifti1Image(np.stack([s.pixel_array for s in slices]), np.eye(4))

    # Load segmentation mask
    mask_image = nib.load(mask_path)
    mask_data = np.rot90(mask_image.get_fdata(), k=1)

    # Display NIfTI volume in sagittal, axial, and coronal views
    volume_data = np.rot90(nifti_image.get_fdata(), k=1)  # Rotate to correct orientation
    print(volume_data.shape)
    volume_data = np.transpose(volume_data, (0, 2, 1))
    print(volume_data.shape)

    # volume_data = volume_data[:, :, -1]

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    sagittal_im = axs[0].imshow(volume_data[int(volume_data.shape[0] / 2), :, :], cmap="gray", origin="lower")
    axs[0].set_title("Sagittal View")
    axial_im = axs[1].imshow(volume_data[:, int(volume_data.shape[1] / 2), :], cmap="gray", origin="lower")
    axs[1].set_title("Axial View")
    coronal_im = axs[2].imshow(volume_data[:, :, int(volume_data.shape[2] / 2)], cmap="gray", origin="lower")
    axs[2].set_title("Coronal View")

    # Display segmentation mask on top of NIfTI volume

    sagittal_mask = axs[0].imshow(mask_data[int(mask_data.shape[0] / 2), :, :], cmap="Reds", alpha=0.3, origin="lower")
    axial_mask = axs[1].imshow(mask_data[:, int(mask_data.shape[1] / 2), :], cmap="Reds", alpha=0.3, origin="lower")
    coronal_mask = axs[2].imshow(mask_data[:, :, int(mask_data.shape[2] / 2)], cmap="Reds", alpha=0.3, origin="lower")

    # Add slider to control slice selection
    slice_axis = 0  # Starting axis
    slice_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Create slider axis
    slice_slider = Slider(slice_slider_ax, 'Slice', 0, mask_data.shape[slice_axis] - 1,
                          valinit=int(mask_data.shape[slice_axis] / 2), valstep=1)  # Create slider

    # Define function to update the slices based on slider value
    def update(val):
        slice_index = int(slice_slider.val)
        if slice_axis == 0:
            sagittal_im.set_data(volume_data[slice_index, :, :])
            axial_im.set_data(volume_data[:, slice_index, :])
            coronal_im.set_data(volume_data[:, :, slice_index])

            sagittal_mask.set_data(mask_data[slice_index, :, :])
            axial_mask.set_data(mask_data[:, slice_index, :])
            coronal_mask.set_data(mask_data[:, :, slice_index])

        elif slice_axis == 1:
            sagittal_im.set_data(volume_data[slice_index, :, :])
            axial_im.set_data(volume_data[:, slice_index, :])
            coronal_im.set_data(volume_data[:, :, slice_index])

            sagittal_mask.set_data(mask_data[slice_index, :, :])
            axial_mask.set_data(mask_data[:, slice_index, :])
            coronal_mask.set_data(mask_data[:, :, slice_index])


        elif slice_axis == 2:
            sagittal_im.set_data(volume_data[slice_index, :, :])
            axial_im.set_data(volume_data[:, slice_index, :])
            coronal_im.set_data(volume_data[:, :, slice_index])

            sagittal_mask.set_data(mask_data[slice_index, :, :])
            axial_mask.set_data(mask_data[:, slice_index, :])
            coronal_mask.set_data(mask_data[:, :, slice_index])

        fig.canvas.draw_idle()

    slice_slider.on_changed(update)

    plt.show()


def display_seg(folder_path):
    """
    Display the image, the segmentation mask and the zoomed tumor
    :param folder_path:
    :return:
    """
    # Read files from folder
    files = os.listdir(folder_path)
    scan_path = None
    mask_path = None

    # Find scan and mask files
    for file in files:
        if 'nii.gz' in file:
            scan_path = os.path.join(folder_path, file)
        elif '.nii' in file:
            mask_path = os.path.join(folder_path, file)

    if scan_path is None or mask_path is None:
        print('Scan or mask file not found')
        return

    # Load scan and mask data
    scan_img = nib.load(scan_path)
    mask_img = nib.load(mask_path)

    scan_data = scan_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Create figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the scan as an image
    scan_slice = scan_data[:, :, scan_data.shape[2] // 2]  # Take a slice from the middle
    scan_im = axs[0].imshow(scan_slice, cmap='gray', origin='lower')

    # Display the mask as an overlay
    mask_im = axs[0].imshow(mask_data[:, :, mask_data.shape[2] // 2], cmap='Reds', alpha=0.3, origin='lower')

    # Set plot title and labels for the scan and mask overlay
    axs[0].set_title('Scan with Mask Overlay')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    # Create a new axis for the zoomed view
    axs[1] = plt.subplot(1, 3, 2)
    axs[1].set_title('Zoomed View')

    # Get the tumor coordinates from the mask
    tumor_coords = mask_data.nonzero()
    min_x, max_x = min(tumor_coords[1]), max(tumor_coords[1])
    min_y, max_y = min(tumor_coords[0]), max(tumor_coords[0])
    tumor_width = max_x - min_x
    tumor_height = max_y - min_y
    center_x = (max_x + min_x) // 2
    center_y = (max_y + min_y) // 2

    # Calculate the zoomed bounding box coordinates
    zoomed_min_x = max(center_x - tumor_width // 2, 0)
    zoomed_max_x = min(center_x + tumor_width // 2, scan_data.shape[1] - 1)
    zoomed_min_y = max(center_y - tumor_height // 2, 0)
    zoomed_max_y = min(center_y + tumor_height // 2, scan_data.shape[0] - 1)

    # Display the zoomed view
    output_size = (224, 224)
    zoomed_scan = scan_data[zoomed_min_y:zoomed_max_y, zoomed_min_x:zoomed_max_x, :]
    # zoomed_scan_resized = resize(zoomed_scan, (224, 224))

    zoomed_scan_resized = ndimage.zoom(zoomed_scan, (output_size[0] / zoomed_scan.shape[0],
                                                     output_size[1] / zoomed_scan.shape[1],
                                                     1), order=3)
    zoomed_im = axs[1].imshow(zoomed_scan_resized[:, :, scan_data.shape[2] // 2], cmap='gray', origin='lower')

    # Set plot title and labels for the zoomed view
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    # Display the zoomed view where the tumor is the largest
    largest_slice_index = np.argmax(np.sum(mask_data, axis=(0, 1)))

    zoomed_scan = scan_data[zoomed_min_y:zoomed_max_y, zoomed_min_x:zoomed_max_x, :]
    # zoomed_scan_resized = resize(zoomed_scan, (224, 224))

    zoomed_scan_resized = ndimage.zoom(zoomed_scan, (output_size[0] / zoomed_scan.shape[0],
                                                     output_size[1] / zoomed_scan.shape[1],
                                                     1), order=3)

    zoomed_slice = zoomed_scan_resized[:, :, largest_slice_index]

    axs[2].imshow(zoomed_slice, cmap='gray', origin='lower')
    # Set plot title and labels for the zoomed view
    axs[2].set_title(largest_slice_index)
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')

    # Create a slider for slice selection
    slice_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Create slider axis
    slice_slider = Slider(slice_slider_ax, 'Slice', 0, scan_data.shape[2] - 1, valinit=scan_data.shape[2] // 2,
                          valstep=1)

    def update_slice(val):
        slice_index = int(slice_slider.val)

        # Update the scan and mask slices
        scan_slice = scan_data[:, :, slice_index]
        mask_slice = mask_data[:, :, slice_index]

        # Update the image data for the scan and mask overlay
        scan_im.set_data(scan_slice)
        mask_im.set_data(mask_slice)

        # Update the zoomed view with the corresponding zoomed slice
        zoomed_slice = zoomed_scan_resized[:, :, slice_index]
        zoomed_im.set_data(zoomed_slice)

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update_slice)
    plt.show()

    return zoomed_slice


def single_input_array(folder_path):
    files = os.listdir(folder_path)
    scan_path = None
    mask_path = None

    # Find scan and mask files
    for file in files:
        if 'nii.gz' in file:
            scan_path = os.path.join(folder_path, file)
        elif '.nii' in file:
            mask_path = os.path.join(folder_path, file)

    if scan_path is None or mask_path is None:
        print('Scan or mask file not found')
        return

    # Load scan and mask data
    scan_img = nib.load(scan_path)
    mask_img = nib.load(mask_path)

    scan_data = scan_img.get_fdata()
    mask_data = mask_img.get_fdata()

    tumor_coords = mask_data.nonzero()
    min_x, max_x = min(tumor_coords[1]), max(tumor_coords[1])
    min_y, max_y = min(tumor_coords[0]), max(tumor_coords[0])
    tumor_width = max_x - min_x
    tumor_height = max_y - min_y
    center_x = (max_x + min_x) // 2
    center_y = (max_y + min_y) // 2

    # Calculate the zoomed bounding box coordinates
    zoomed_min_x = max(center_x - tumor_width // 2, 0)
    zoomed_max_x = min(center_x + tumor_width // 2, scan_data.shape[1] - 1)
    zoomed_min_y = max(center_y - tumor_height // 2, 0)
    zoomed_max_y = min(center_y + tumor_height // 2, scan_data.shape[0] - 1)

    # Display the zoomed view where the tumor is the largest
    output_size = (224, 224)
    largest_slice_index = np.argmax(np.sum(mask_data, axis=(0, 1)))

    zoomed_scan = scan_data[zoomed_min_y:zoomed_max_y, zoomed_min_x:zoomed_max_x, :]
    zoomed_scan_resized = ndimage.zoom(zoomed_scan, (output_size[0] / zoomed_scan.shape[0],
                                                     output_size[1] / zoomed_scan.shape[1],
                                                     1), order=3)
    zoomed_slice = zoomed_scan_resized[:, :, largest_slice_index]

    return zoomed_slice


def multi_input_array(folder_path):
    """
    Creates 224x224 tumor images from .nii files.
    :param folder_path: path to segmentation and scan folder
    :return: zoomed_slices, patient_id
    """
    files = os.listdir(folder_path)
    scan_path = None
    mask_path = None

    # Find scan and mask files
    for file in files:
        if 'nii.gz' in file:
            scan_path = os.path.join(folder_path, file)
        elif '.nii' in file:
            mask_path = os.path.join(folder_path, file)

    if scan_path is None or mask_path is None:
        print('Scan or mask file not found')
        return None, None

    # Load scan and mask data
    scan_img = nib.load(scan_path)
    mask_img = nib.load(mask_path)

    scan_data = scan_img.get_fdata()
    mask_data = mask_img.get_fdata()

    tumor_coords = mask_data.nonzero()
    min_x, max_x = min(tumor_coords[1]), max(tumor_coords[1])
    min_y, max_y = min(tumor_coords[0]), max(tumor_coords[0])
    tumor_width = max_x - min_x
    tumor_height = max_y - min_y
    center_x = (max_x + min_x) // 2
    center_y = (max_y + min_y) // 2

    # Calculate the zoomed bounding box coordinates
    zoomed_min_x = max(center_x - tumor_width // 2, 0)
    zoomed_max_x = min(center_x + tumor_width // 2, scan_data.shape[1] - 1)
    zoomed_min_y = max(center_y - tumor_height // 2, 0)
    zoomed_max_y = min(center_y + tumor_height // 2, scan_data.shape[0] - 1)

    zoomed_slices = []
    output_size = (224, 224)

    for slice_index in range(scan_data.shape[2]):
        if np.any(mask_data[:, :, slice_index]):
            zoomed_slice = scan_data[zoomed_min_y:zoomed_max_y, zoomed_min_x:zoomed_max_x, slice_index]
            # Apply interpolation if needed
            zoomed_slice_resized = cv2.resize(zoomed_slice, output_size)
            zoomed_slices.append(zoomed_slice_resized)

    # Extract patient ID from scan file name
    file_name = os.path.basename(scan_path)
    patient_id = file_name[:9]  # Extract the first 9 characters

    return zoomed_slices, patient_id




def save_array_to_hdf5(array, filename):
    """
    Save the image with the largest tumor to an hdf5 file
    :param array:
    :param filename:
    :return:
    """
    with h5py.File(filename, 'w') as f:
        # Create a group called 'patient'
        group = f.create_group('Patient')
        f_array = array.astype(np.float32)
        # Create a variable named 'train' within the 'patient' group

        min_val = np.min(f_array)
        max_val = np.max(f_array)
        scaled_arr = (f_array - min_val) / (max_val - min_val)  # Scale between 0 and 1
        scaled_arr = 2 * scaled_arr - 1

        group['train'] = scaled_arr

        # Create an empty variable named 'target' (type: long) within the 'patient' group
        group['target'] = 1


def save_arrays_to_hdf5(array, patient_id): xd
    """
    Save all the tumor images in an HDF5 file.
    :param array: List or array of tumor images (output of multi_input_array)
    :param patient_id: Patient ID
    :return: None
    """
    save_location = '/home/adamdiakite/Documents/ESPS-main/Test'

    folder_path = os.path.join(save_location, 'patient_data')
    os.makedirs(folder_path, exist_ok=True)  # Creates the patient_data folder if it doesn't exist

    filename = os.path.join(folder_path, f'{patient_id}.hdf5')

    with h5py.File(filename, 'w') as f:
        # Create a dataset for the patient ID

        for i, element in enumerate(array):
            slice_group = f.create_group(f'slice_{i}')

            f_array = array[i].astype(np.float32)
            min_val = np.min(f_array)
            max_val = np.max(f_array)
            scaled_arr = (f_array - min_val) / (max_val - min_val)  # Scale between 0 and 1
            scaled_arr = 2 * scaled_arr - 1
            scaled_arr = np.where(scaled_arr > 0, 1, 0)

            slice_group.create_dataset('train', data=scaled_arr, track_order=True)
            slice_group.create_dataset('target', shape=(), dtype='int64', data=1)

            # Calculate tumor area
            tumor_area = np.sum(scaled_arr > 0)  # Count the number of elements above 0
            slice_group.create_dataset('tumor_area', shape=(), dtype='int64', data=tumor_area)

    print("Arrays saved to HDF5 file.")


array, patient_id = multi_input_array(path)
save_arrays_to_hdf5(array, patient_id)

