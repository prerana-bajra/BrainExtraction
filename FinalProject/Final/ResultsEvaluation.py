import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os


def evaluate_brain_extraction(true_mask, predicted_mask):
    # Convert the masks to binary
    true_mask = np.asarray(true_mask, dtype=bool)
    predicted_mask = np.asarray(predicted_mask, dtype=bool)

    # Calculate true positive, false positive, false negative
    true_positive = np.logical_and(true_mask, predicted_mask).sum()
    false_positive = np.logical_and(~true_mask, predicted_mask).sum()
    false_negative = np.logical_and(true_mask, ~predicted_mask).sum()

    # Calculate Dice coefficient
    dice_coefficient = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative)

    # Calculate Jaccard similarity coefficient
    jaccard_similarity = true_positive / (true_positive + false_positive + false_negative)

    # Calculate sensitivity (true positive rate)
    sensitivity = true_positive / (true_positive + false_negative)

    # Calculate specificity (true negative rate)
    specificity = np.sum(~true_mask & ~predicted_mask) / np.sum(~true_mask)

    # Calculate accuracy
    accuracy = (true_positive + np.sum(~true_mask & ~predicted_mask)) / (
                true_positive + false_positive + false_negative + np.sum(~true_mask))

    return dice_coefficient, jaccard_similarity, sensitivity, specificity, accuracy


def load_predicted_mask(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    # Get the data array from the image
    data = img.get_fdata()
    # Threshold the data to obtain the binary mask
    # (assuming the brain regions have intensity values of 1 and background is 0)
    predicted_mask = data > 0
    return predicted_mask


def load_mgz(file_path):
    # file_path = "images/brainmask_09.auto.mgz"

    # Load the MGZ image
    img = nib.load(file_path)

    # Get the image data as a numpy array
    data = img.get_fdata()

    true_mask = data > 0
    return true_mask

    # Example: Display the image slice at index 0 in the sagittal plane

    # plt.imshow(data[:, :, 192].T, cmap="gray")
    # plt.show()


def get_file_path():

    # Source directory where the files are located
    source_directory = 'Dataset/ds000113-download/T1'

    # Destination directory where the corresponding files are located
    destination_directory = "Dataset/ds000113-download/T2/freesurfer"

    # List of files for which you want to find corresponding files
    file_list = [file for file in os.listdir(source_directory) if file.endswith('.nii.gz')]
    print(file_list)

    dice_coefficient_array = np.array([])
    jaccard_similarity_array = np.array([])
    sensitivity_array = np.array([])
    specificity_array = np.array([])
    accuracy_array = np.array([])

    # Iterate over the files in the file list
    for file_name in file_list:

        input_file_path = os.path.join(source_directory, file_name)
        # Extract the file number from the file name
        file_number = file_name.split("-")[1].split(".")[0].replace('_ses', '')
        # print(file_number)
        # Construct the corresponding file name
        corresponding_file_name = f"brainmask_{file_number}.auto.mgz"

        # Construct the full path to the corresponding file in the destination directory
        corresponding_file_path = os.path.join(destination_directory, corresponding_file_name)

        # Check if the corresponding file exists
        if os.path.isfile(corresponding_file_path):
            print(f"Corresponding file found: {corresponding_file_path}")
            freesurfer_mask = load_mgz(corresponding_file_path)
            predicted_mask = load_predicted_mask(input_file_path)

            dice_coefficient, jaccard_similarity, sensitivity, specificity, accuracy = evaluate_brain_extraction(freesurfer_mask, predicted_mask)

            print('dice_coefficient', dice_coefficient)
            print('jaccard_similarity', jaccard_similarity)
            print('sensitivity', sensitivity)
            print('specificity', specificity)
            print('accuracy', accuracy)

            dice_coefficient_array = np.append(dice_coefficient_array, dice_coefficient)
            jaccard_similarity_array = np.append(jaccard_similarity_array, jaccard_similarity)
            sensitivity_array = np.append(sensitivity_array, sensitivity)
            specificity_array = np.append(specificity_array, specificity)
            accuracy_array = np.append(accuracy_array, accuracy)

        else:
            print(f"No corresponding file found for: {file_name}")

        print('max dice_coefficient', np.max(dice_coefficient_array))
        print('max jaccard_similarity', np.max(jaccard_similarity_array))
        print('max sensitivity', np.max(sensitivity_array))
        print('max specificity', np.max(specificity_array))
        print('max accuracy', np.max(accuracy_array))


if __name__ == '__main__':
    get_file_path()
