from skimage import filters, morphology
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2
import os


def view_img(img_path, label):
    # Load the NIfTI image
    img = nib.load(img_path)

    # Get the image data and shape
    data = img.get_fdata()
    # view_data(data, label)


def view_data(data, label):
    shape = data.shape

    # Define the number of frames to display
    n_frames = 6

    # Calculate the step size to get approximately 'n_frames' frames
    step = shape[2] // n_frames

    # Define the number of columns to display the plots
    n_cols = 3

    # Calculate the number of rows needed
    n_rows = n_frames // n_cols

    # Create a figure and axis object
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(8, 5))

    # Loop through the frames and display them
    for i in range(n_frames):
        # Get the current row and column indices
        row = i // n_cols
        col = i % n_cols

        # Get the current frame
        frame = data[:, :, i * step]

        # Display the frame
        ax[row, col].imshow(frame, cmap='gray')
        ax[row, col].set_title('Frame {}'.format(i * step))
        ax[row, col].axis('off')

    fig.suptitle(label)

    # Show the plot
    plt.show()


def load_nifti(img_path):
    img_file = img_path
    img = nib.load(img_file)
    img_data = img.get_fdata()
    return img, img_data


def plt_plot(data):
    plt.imshow(data)
    plt.show()


def gauss_morph(sigma, radius_erosion, radius_dilation, data):
    # Apply Gaussian smoothing filter
    smooth_data = filters.gaussian(data, sigma=sigma)

    # Apply intensity normalization
    norm_data = (smooth_data - np.min(smooth_data)) / (np.max(smooth_data) - np.min(smooth_data))

    # Threshold the image to extract brain tissue
    threshold = filters.threshold_otsu(norm_data)

    mask = norm_data > threshold
    # view_data(mask, 'otsu')
    # Apply morphological operations to remove non-brain tissue
    mask = morphology.erosion(mask, morphology.ball(radius_erosion))
    # view_data(mask, 'erosion')

    # mask = morphology.dilation(mask, morphology.ball(radius_dilation))
    # view_data(mask, 'dilation')

    # Apply the mask to the original image
    # brain_data = data * mask
    # Save the preprocessed image
    return mask


def llc(sigma, radius_erosion, radius_dilation, data):
    # Apply Gaussian smoothing filter
    # smooth_data = filters.gaussian(data, sigma=sigma)
    smooth_data = data
    # Apply intensity normalization
    norm_data = (smooth_data - np.min(smooth_data)) / (np.max(smooth_data) - np.min(smooth_data))
    # view_data(norm_data, 'norm')

    # Threshold the image to extract brain tissue
    threshold = filters.threshold_otsu(norm_data)
    # Display the thresholded image
    print(threshold)
    mask = norm_data > threshold
    mask = morphology.erosion(mask, morphology.ball(radius_erosion))

    # view_data(mask, 'After erosion mask')

    # Label the connected components in the mask
    labeled_data, label = ndi.label(mask)
    print(label)
    # Calculate the size of each connected component
    sizes = np.zeros(2, dtype=np.int32)
    for i in range(1, 2):
        sizes[i] = np.sum(labeled_data == i)
    print(sizes)
    # Find the label of the largest connected component
    largest_label = np.argmax(sizes)
    print(largest_label)
    # Create a binary mask that includes only the voxels belonging to the largest component
    largest_mask = (labeled_data == largest_label)
    # view_data(largest_mask, 'largest_mask')

    largest_mask = morphology.dilation(largest_mask, morphology.ball(radius_dilation))
    # view_data(largest_mask, 'dilation largest_mask')
    largest_mask = morphology.dilation(largest_mask, morphology.ball(radius_dilation))
    # view_data(largest_mask, 'dilation largest_mask')
    return largest_mask


def brain_ext(input_dir, output_dir):
    # Get a list of NIfTI files in the input directory
    file_list = [file for file in os.listdir(input_dir) if file.endswith('.nii.gz')]
    print(file_list)

    i = 0
    # Loop through the file list
    for file_name in file_list:
        # Load the NIfTI file
        file_path = os.path.join(input_dir, file_name)

        # first(file_path)
        # second()

        img, img_data = load_nifti(file_path)
        data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # view_img(file_path, 'Input')
        mask_1 = gauss_morph(3, 3, 3, data)
        brain_data = mask_1 * data

        largest_mask = llc(5, 5, 5, brain_data)
        segmented_data = brain_data.copy()
        segmented_data[~largest_mask] = 0
        # view_data(segmented_data, 'segmented')

        # Create a new NIfTI image with the processed data
        processed_img = nib.Nifti1Image(segmented_data, img.affine, img.header)

        # Save the processed NIfTI image to the output directory
        output_path = os.path.join(output_dir, "segmented_" + file_name)
        print(output_path)
        nib.save(processed_img, output_path)

        i += 1
        print("done: - ", i)


if __name__ == '__main__':
    # # Directory containing the NIfTI files
    # input_dir = 'Dataset/ds000113-download/T2'
    #
    # # Directory to save the processed files
    # output_dir = 'Output/'

    # Directory containing the NIfTI files
    input_dir = 'Dataset/ds000113-download/T1'

    # Directory to save the processed files
    output_dir = 'Output/T1/'

    brain_ext(input_dir, output_dir)


