"""
This script process 3DIRCA dataset. The end result is some niftis with only liver and lesion labels  (e.g., label01.nii)
Label values: 0,1,2 for bg,liver,lesion resp.

Usage :
cd into the directory of the IRCA dataset (usually folder named 3Dircadb1)
then run the script :
python /path/to/irca_to_nii.python


"""
from matplotlib import pyplot as plt
#%matplotlib inline
import dicom
#import medpy
import nibabel
import numpy as np
import os, sys, glob, re
import natsort

MAX_VOLUMES = -1

def flip_volume(input_filename, output_filename):
    """ Flips left to right, a given volume and writes it to output_filename.
    This is necessary for irca image.nii's to have the same orientation as fire3 database """
    print "\t Flipping left-right", input_filename,'to',output_filename
    volume = nibabel.load(input_filename).get_data()
    volume = np.fliplr(volume)
    nii = nibabel.Nifti1Image(volume, affine = np.eye(4))
    nii.to_filename(output_filename)

def read_dicom_series(directory, filepattern = "image_*"):
    """ Reads a DICOM Series files in the given directory. Only filesnames matching filepattern will be considered"""
    
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : "+str(directory))
    print '\t\t Read Dicom',directory
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print '\t\t Length dicom series',len(lstFilesDCM)
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom

# Path to root folder of dataset : 3Dircadb1
DATA_PATH = '.'
# path that will contain the output niftis
OUTPUT_PATH = './niftis_segmented_all/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    
liver_volumes = []
tumor_volumes = []
volume_ids = []
#######################
##### READ DICOMS #####
#######################

for i, volume_dirname in enumerate(natsort.natsorted(os.listdir(DATA_PATH))):
    if MAX_VOLUMES > -1 and i+1 > MAX_VOLUMES:
        break
    # Get full dirname : /path/to/3Dircadb1.5/
    volume_fulldirname = os.path.join(DATA_PATH,volume_dirname)
    # Make sure it's really a volume directory (like 3Dircadb1.5/ )
    if not os.path.isdir(volume_fulldirname) or not volume_dirname.startswith("3Dircadb1."):
        continue
    print 'Volume',volume_dirname
    relevant_masks = 0 # number of relevant masks files found for this volume (liver or tumor)
    mask_dirname = os.path.join(volume_fulldirname,"MASKS_DICOM")
    volume_id = volume_dirname.replace("3Dircadb1.","") #id of volume ("1" to "20")
    # Save the volume id
    volume_ids.append(int(volume_id))
    image_filename = os.path.join(volume_fulldirname,"image"+volume_id+".nii")
    flip_volume(image_filename, os.path.join(OUTPUT_PATH, "image%.2d"%int(volume_id)+".nii"))
    
    tumor_volume = None
    liver_volume = None
    
    # For each relevant organ in the current volume
    for organ in os.listdir(mask_dirname):
        organ_path = os.path.join(mask_dirname,organ)
        if not os.path.isdir(organ_path):
            continue
        
        organ = organ.lower()
        
        if organ.startswith("livertumor") or re.match("liver.yst.*", organ) or organ.startswith("stone") or organ.startswith("metastasecto") :
            print '\tOrgan',mask_dirname,organ
            current_tumor = read_dicom_series(organ_path)
            current_tumor = np.clip(current_tumor,0,1)
            if tumor_volume is None:
                tumor_volume = current_tumor
            else:
                # Merge tumors
                tumor_volume = np.logical_or(tumor_volume,current_tumor)

            relevant_masks+=1
        elif organ == 'liver':
            if liver_volume is not None:
                print 'WARNING:',"Liver volume was already read! Another liver was found now at %s" % organ_path
            print '\tOrgan',mask_dirname,organ
            liver_volume = read_dicom_series(organ_path)
            liver_volume = np.clip(liver_volume, 0, 1)
            relevant_masks+=1
    
    #########################
    ##### Special Cases #####
    #
    #
    if volume_dirname == "3Dircadb1.3":
        print '\tHandling Special case for volume 3. Merging label 329 from LABELED_DICOM'
        ### Volume 3. Need to merge a label from the LABELED_DICOM into tumors
        label_dicom_path = os.path.join(volume_fulldirname, "LABELLED_DICOM") 
        desired_label_value = 329
        extra_mask = read_dicom_series(label_dicom_path) == 329
        tumor_volume[extra_mask] = 1
    #
    #
    #########################
        
    liver_volumes.append(liver_volume)
    tumor_volumes.append(tumor_volume)
    if relevant_masks < 2:
        print "For %s Only %i relevant mask(s) were found" % (volume_dirname,relevant_masks)








print ' WRITING NIFTIS TO DISK'
#######################
##### WRITE NIFTI #####
#######################

#final_volume = []
for i in range(len(liver_volumes)):
    liver, tumor, volume_id = liver_volumes[i], tumor_volumes[i], volume_ids[i]
    # Map label values to 0 and 1 and merge all into liver volume
    liver_value = np.max(np.unique(liver))
    liver[liver==liver_value] = 1
    
    if tumor is not None:
        # Because different files have different types (True/False or 0/1 or 0/255)
        # We get the actual tumor value here
        tumor_value = np.max(np.unique(tumor))
        # Merge into liver volume
        liver[tumor==tumor_value] = 2

    #final_volume.append(liver)
    # Write to disk
    nii=nibabel.Nifti1Image(np.rot90(liver, -1), affine=np.eye(4))
    filename = os.path.join(OUTPUT_PATH, "label%.2d.nii" % (volume_id))
    print 'Writing to file ',filename
    nii.to_filename(filename)
    
    
    
    