import nibabel as nib
# import fslpy


# Load T2-weighted MRI data
t2_image = nib.load('Neurohacking_data-0.0/BRAINIX/NIfTI/BRAINIX_NIFTI_T2.nii.gz')
print(t2_image)
# # Create FSL-BET object
# bet = fslpy.BET()
#
# # Set BET parameters
# bet.inputs.in_file = t2_image.get_filename()
# bet.inputs.frac = 0.5
#
# # Run BET
# bet.run()
#
# # Save brain-extracted image and mask
# nib.save(nib.load(bet.outputs.out_file), 'Output/t2_brain.nii.gz')
# nib.save(nib.load(bet.outputs.mask_file), 'Output/t2_mask.nii.gz')
