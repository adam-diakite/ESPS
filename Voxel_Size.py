import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def compare_voxel_sizes(file1_path, file2_path):
    # Load the NIfTI files
    img1 = nib.load(file1_path)
    img2 = nib.load(file2_path)

    # Get the voxel sizes
    voxel_size1 = img1.header.get_zooms()
    voxel_size2 = img2.header.get_zooms()

    # Print the voxel sizes
    print("Voxel size of file 1:", voxel_size1)
    print("Voxel size of file 2:", voxel_size2)


def calculate_tumor_surface(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Threshold the data to isolate the tumor
    tumor_mask = data > 0

    # Calculate the tumor surface by summing the number of boundary voxels
    dx, dy, dz = img.header.get_zooms()
    surface = np.sum(np.abs(np.gradient(tumor_mask.astype(float), dx, dy, dz)) > 0)

    return surface



def plot_middle_slice(file1_path, file2_path):
    # Load the NIfTI files
    img1 = nib.load(file1_path)
    img2 = nib.load(file2_path)

    # Get the data arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Calculate the middle slice index
    mid_slice_index1 = data1.shape[-1] // 2
    mid_slice_index2 = data2.shape[-1] // 2

    # Extract the middle slices
    mid_slice1 = data1[..., mid_slice_index1]
    mid_slice2 = data2[..., mid_slice_index2]

    # Plot the middle slices
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mid_slice1, cmap='gray')
    plt.title('File 1 Middle Slice')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mid_slice2, cmap='gray')
    plt.title('File 2 Middle Slice')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


file1_path = "/home/adamdiakite/Bureau/testnii/scan0021/2-21-0021_2-21-0021_tapc_iv_2-21-0021_(2-21-0021)_5_tzyx_float32_HU_na5_nt1_nz587_ny512_nx512.nii.gz"
file2_path = "/home/adamdiakite/Bureau/testnii/scan0021/Segmentation_6.nii"

compare_voxel_sizes(file1_path, file2_path)
plot_middle_slice(file1_path, file2_path)

# compare_voxel_sizes("/home/adamdiakite/Bureau/testnii/scans0046//2-21-0046_2-21-0046_Scanner_thoracique_ave_2-21-0046_(2-21-0046)_2_tzyx_float32_HU_na1_nt1_nz621_ny512_nx512.nii.gz", "/home/adamdiakite/Bureau/testnii/scans0046/Segmentation.nii")
