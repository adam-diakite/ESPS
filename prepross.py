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
from sklearn import preprocessing

root_folder = '/media/adamdiakite/LaCie/database_paris'
save_folder = '/media/adamdiakite/LaCie/save'


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
    Display the image, the segmentation mask, and the zoomed tumor
    :param folder_path: Path to the folder containing the scan and mask files
    :return: None
    """
    # Read files from folder
    files = os.listdir(folder_path)
    scan_path = None
    mask_path = None

    # Find scan and mask files
    for file in files:
        if 'segmentation' in file.lower() or file.endswith('.nii'):
            mask_path = os.path.join(folder_path, file)
        else:
            scan_path = os.path.join(folder_path, file)

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
    scan_slice = np.rot90(scan_data[:, :, scan_data.shape[2] // 2])  # Rotate the slice clockwise
    scan_im = axs[0].imshow(scan_slice, cmap='gray', origin='lower')

    # Display the mask as an overlay
    mask_im = axs[0].imshow(np.rot90(mask_data[:, :, mask_data.shape[2] // 2]), cmap='Reds', alpha=0.3, origin='lower')

    # Set plot title and labels for the scan and mask overlay
    axs[0].set_title('Scan with Mask Overlay')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    # Create a new axis for the zoomed view
    axs[1] = plt.subplot(1, 2, 2)
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
    zoomed_scan_resized = ndimage.zoom(zoomed_scan, (output_size[0] / zoomed_scan.shape[0],
                                                     output_size[1] / zoomed_scan.shape[1],
                                                     1), order=3)
    zoomed_im = axs[1].imshow(np.rot90(zoomed_scan_resized[:, :, scan_data.shape[2] // 2]), cmap='gray', origin='lower')

    # Set plot title and labels for the zoomed view
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    # Display the zoomed view where the tumor is the largest
    largest_slice_index = np.argmax(np.sum(mask_data, axis=(0, 1)))

    zoomed_scan = scan_data[zoomed_min_y:zoomed_max_y, zoomed_min_x:zoomed_max_x, :]
    zoomed_scan_resized = ndimage.zoom(zoomed_scan, (output_size[0] / zoomed_scan.shape[0],
                                                     output_size[1] / zoomed_scan.shape[1],
                                                     1), order=3)

    zoomed_slice = np.rot90(zoomed_scan_resized[:, :, largest_slice_index])  # Rotate the slice clockwise

    axs[1].imshow(zoomed_slice, cmap='gray', origin='lower')
    # Set plot title and labels for the zoomed view
    axs[1].set_title(largest_slice_index)
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    def update_slice(val):
        # Get the current slider value
        slice_index = int(slice_slider.val)

        # Update the scan slice and zoomed slice
        scan_slice = np.rot90(scan_data[:, :, slice_index])
        zoomed_slice = np.rot90(zoomed_scan_resized[:, :, slice_index])

        # Update the image data
        scan_im.set_data(scan_slice)
        zoomed_im.set_data(zoomed_slice)

        # Set plot title for the zoomed view
        axs[1].set_title(largest_slice_index)

        # Redraw the figure
        fig.canvas.draw()

    # Connect the update_slice function to the slider's on_changed event
    slice_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Create slider axis
    slice_slider = Slider(slice_slider_ax, 'Slice', 0, scan_data.shape[2] - 1, valinit=scan_data.shape[2] // 2,
                          valstep=1)
    slice_slider.on_changed(update_slice)

    # Show the plot
    plt.show()

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


def resize_image(image, target_size):
    """
    Resizes the image while maintaining the aspect ratio.
    :param image: input image
    :param target_size: target size (int)
    :return: resized image
    """
    height, width = image.shape[:2]

    if len(image.shape) == 3:
        # Color image (3 channels)
        if height > width:
            aspect_ratio = float(target_size) / height
            new_height = target_size
            new_width = int(width * aspect_ratio)
        else:
            aspect_ratio = float(target_size) / width
            new_height = int(height * aspect_ratio)
            new_width = target_size

        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        # Grayscale image (1 channel)
        aspect_ratio = float(target_size) / max(height, width)
        new_height = int(height * aspect_ratio)
        new_width = int(width * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_image


def multi_input_array(folder_path):
    """
    Creates 224x224 tumor images from .nii files.
    :param folder_path: path to segmentation and scan folder
    :return: zoomed_slices, tumor_area, patient_id
    """
    files = os.listdir(folder_path)
    scan_path = None
    mask_path = None
    desired_shape = 224

    # Find scan and mask files
    for file in files:
        if 'segmentation' in file.lower() or file.endswith('.nii'):
            mask_path = os.path.join(folder_path, file)
        else:
            scan_path = os.path.join(folder_path, file)

    if scan_path is None or mask_path is None:
        print('Scan or mask file not found')
        return None, None, None

    # Load scan and mask data
    scan_img = nib.load(scan_path)
    mask_img = nib.load(mask_path)

    scan_data = scan_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Change the voxel size of the mask to 1x1x1
    mask_header = mask_img.header.copy()
    mask_header.set_zooms((1, 1, 1))

    # Coordonnées de la tumeur en se servant du masque
    tumor_coords = mask_data.nonzero()
    min_x, max_x = min(tumor_coords[1]), max(tumor_coords[1])
    min_y, max_y = min(tumor_coords[0]), max(tumor_coords[0])
    tumor_width = max_x - min_x
    tumor_height = max_y - min_y
    center_x = (max_x + min_x) // 2
    center_y = (max_y + min_y) // 2

    # Calculate the zoomed bounding box coordinates
    zoomed_min_x = max(center_x - tumor_width // 2 - 10, 0)
    zoomed_max_x = min(center_x + tumor_width // 2 + 10, scan_data.shape[1] - 1)
    zoomed_min_y = max(center_y - tumor_height // 2 - 10, 0)
    zoomed_max_y = min(center_y + tumor_height // 2 + 10, scan_data.shape[0] - 1)

    # Calculate the required padding to achieve a square shape of 224x224
    current_width = zoomed_max_x - zoomed_min_x
    current_height = zoomed_max_y - zoomed_min_y

    if current_width > current_height:
        diff = current_width - current_height
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        zoomed_min_y = max(zoomed_min_y - pad_top, 0)
        zoomed_max_y = min(zoomed_max_y + pad_bottom, scan_data.shape[0] - 1)
    elif current_height > current_width:
        diff = current_height - current_width
        pad_left = diff // 2
        pad_right = diff - pad_left
        zoomed_min_x = max(zoomed_min_x - pad_left, 0)
        zoomed_max_x = min(zoomed_max_x + pad_right, scan_data.shape[1] - 1)

    zoomed_slices = []
    zoomed_tumors = []
    tumor_area = []

    for slice_index in range(scan_data.shape[2]):
        if np.any(mask_data[:, :, slice_index]):
            zoomed_slice = scan_data[zoomed_min_y:zoomed_max_y, zoomed_min_x:zoomed_max_x, slice_index]
            try:
                zoomed_slice_resized = cv2.resize(zoomed_slice, (desired_shape, desired_shape))
                zoomed_slice_resized = cv2.rotate(zoomed_slice_resized, cv2.ROTATE_90_CLOCKWISE)
                zoomed_slice_resized = cv2.flip(zoomed_slice_resized, 1)
                zoomed_slices.append(zoomed_slice_resized)
                tumor_area.append(np.sum(mask_data[:, :, slice_index]))
            except cv2.error as e:
                print(f"Error occurred during resizing: {e}. Skipping patient.")
                continue

    zoomed_slices = zoomed_slices[::-1]  # reversing using list slicing

    # Extract patient ID from scan file name
    file_name = os.path.basename(scan_path)
    patient_id = file_name[:9]  # Extract the first 9 characters
    print('Patient ID:', patient_id)

    # # Display zoomed slices
    # num_slices = len(zoomed_slices)
    # num_cols = 5  # Number of columns in the plot
    # num_rows = (num_slices + num_cols - 1) // num_cols  # Calculate number of rows
    #
    # plt.figure(figsize=(15, 3 * num_rows))  # Set figure size
    #
    # for i, slice_img in enumerate(zoomed_slices):
    #     plt.subplot(num_rows, num_cols, i + 1)
    #     plt.imshow(slice_img, cmap='gray')
    #     plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return zoomed_slices, patient_id, tumor_area





def save_arrays_to_hdf5(array, patient_id, tumor_area):
    """
    Save all the tumor images in an HDF5 file and plot histograms.
    :param array: List or array of tumor images (output of multi_input_array)
    :param tumor_area: List or array of tumor areas corresponding to each slice
    :param patient_id: Patient ID
    :return: None
    """
    save_location = '/home/adamdiakite/Documents/ESPS-main/HDF5_Test'

    folder_path = os.path.join(save_location, 'patient_data_uniform')
    os.makedirs(folder_path, exist_ok=True)  # Creates the patient_data folder if it doesn't exist

    filename = os.path.join(folder_path, f'{patient_id}.hdf5')

    try:
        with h5py.File(filename, 'w') as f:
            # Create a dataset for the patient ID

            for i, element in enumerate(array):
                try:
                    slice_group = f.create_group(f'slice_{i}')

                    # Rotate the image clockwise
                    rotated_image = cv2.rotate(array[i], cv2.ROTATE_90_CLOCKWISE)

                    # Image scan
                    f_array = rotated_image.astype(np.float32)

                    # Set values greater than 700 to 700
                    f_array[f_array > 700] = 700

                    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
                    ascolumns = f_array.reshape(-1, 1)
                    t = min_max_scaler.fit_transform(ascolumns)
                    scaled_arr = t.reshape(f_array.shape)

                    # Randomize the indices of the flattened array
                    flattened_indices = np.arange(scaled_arr.size)
                    np.random.shuffle(flattened_indices)

                    # Reshape the randomized indices to match the array shape
                    randomized_indices = np.unravel_index(flattened_indices, scaled_arr.shape)

                    # Create a new array with randomized indices
                    randomized_arr = np.zeros_like(scaled_arr)
                    randomized_arr[randomized_indices] = scaled_arr.flatten()

                    # Group creation in HDF5
                    slice_group.create_dataset('train', data=randomized_arr, track_order=True)
                    slice_group.create_dataset('target', shape=(), dtype='int64', data=1)

                    # Get tumor area from the provided tumor_area parameter
                    tumor_area_value = tumor_area[i]
                    slice_group.create_dataset('tumor_area', shape=(), dtype='int64', data=tumor_area_value)

                    if i == 28 or i == 32:
                        plt.hist(randomized_arr.flatten(), bins=50, alpha=0.5, label=f'Slice {i}')
                except IndexError as e:
                    print(f"IndexError encountered while processing slice {i}: {e}")
                    print("Skipping to the next slice.")
                    continue

        # Plot the histograms
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Pixel Values')
        plt.legend()
        plt.show()

        print("Arrays saved to HDF5 file.")
    except EOFError as e:
        print(f"Error encountered while saving HDF5 file: {e}")
        print("Skipping to the next file.")

def process_scan_folder(root_folder):
    """
    Process only 'scans' folders in an entire dataset.
    :param root_folder: Root folder of the dataset
    :return: None
    """
    for root, dirs, files in os.walk(root_folder):
        for folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            if folder_name.lower() == 'scans':
                # Process 'scans' subfolder
                result = multi_input_array(folder_path)
                if result is not None:
                    array, patient_id, tumor_area = result
                    save_arrays_to_hdf5(array, patient_id, tumor_area)


process_scan_folder('/media/adamdiakite/LaCie/patient49')

# display_seg("/media/adamdiakite/LaCie/database_paris/2-21-0059-2-21-0059/20081229-THO ABDOPELV CR STD/3-TAP Portal/scans")
