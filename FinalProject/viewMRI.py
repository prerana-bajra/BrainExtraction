import nibabel as nib
import matplotlib.pyplot as plt

# Load NIfTI image
nii_img = nib.load('Neurohacking_data-0.0/BRAINIX/NIfTI/BRAINIX_NIFTI_T2.nii.gz')

# Get image data and header
data = nii_img.get_fdata()
header = nii_img.header

# Plot image
plt.imshow(data[:, :, 10], cmap='gray')
plt.show()
