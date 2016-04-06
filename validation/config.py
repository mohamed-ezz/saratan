import logging

# Number of CPUs used for parallel processing
N_PROC = 14

# Image/Seg shape
slice_shape = (388,388)

#################
#### OUTPUT #####
#################

# Logging level
log_level = logging.INFO
output_dir = "."
logfile = 'output.txt'
# Save liver.npy and lesion.npy volumes to output_dir/[niftiname].liver.npy, with shape h,w,slices,classes
save_probability_volumes = True
# Save slices as png files. This param is the increment between plotting one slice and the next
# Set 0 or -1 to disable plotting
plot_every_n_slices = -1


###########################
##### 3DIRCA DATASET ######
###########################

#Use this line on IBBM computers
IRCA_BASE_PATH = '/media/nas/01_Datasets/CT/Abdomen/3Dircadb1/niftis_segmented_lesions/'
#Use the Following line for slu02
#IRCA_BASE_PATH = '/data/niftis_segmented/'

#the array after the label element is the voxel spacing
irca_all= [\
(301,IRCA_BASE_PATH+"image01.nii",IRCA_BASE_PATH+"label01.nii",[0.57,0.57,1.6]),
(302,IRCA_BASE_PATH+"image02.nii",IRCA_BASE_PATH+"label02.nii",[0.78,0.78,1.6]),
(303,IRCA_BASE_PATH+"image03.nii",IRCA_BASE_PATH+"label03.nii",[0.62,0.62,1.25]),
(304,IRCA_BASE_PATH+"image04.nii",IRCA_BASE_PATH+"label04.nii",[0.74,0.74,2.]),
#(305,IRCA_BASE_PATH+"image05.nii",IRCA_BASE_PATH+"label05.nii",[0.78,0.78,1.6]),
(306,IRCA_BASE_PATH+"image06.nii",IRCA_BASE_PATH+"label06.nii",[0.78,0.78,1.6]),
#(307,IRCA_BASE_PATH+"image07.nii",IRCA_BASE_PATH+"label07.nii",[0.78,0.78,1.6]),
(308,IRCA_BASE_PATH+"image08.nii",IRCA_BASE_PATH+"label08.nii",[0.56,0.56,1.6]),
(309,IRCA_BASE_PATH+"image09.nii",IRCA_BASE_PATH+"label09.nii",[0.87,0.87,2.]),
(310,IRCA_BASE_PATH+"image10.nii",IRCA_BASE_PATH+"label10.nii",[0.73,0.73,1.6]),
#(311,IRCA_BASE_PATH+"image11.nii",IRCA_BASE_PATH+"label11.nii",[0.72,0.72,1.6]),
(312,IRCA_BASE_PATH+"image12.nii",IRCA_BASE_PATH+"label12.nii",[0.68,0.68,1.]),
(313,IRCA_BASE_PATH+"image13.nii",IRCA_BASE_PATH+"label13.nii",[0.67,0.67,1.6]),
#(314,IRCA_BASE_PATH+"image14.nii",IRCA_BASE_PATH+"label14.nii",[0.72,0.72,1.6]),
(315,IRCA_BASE_PATH+"image15.nii",IRCA_BASE_PATH+"label15.nii",[0.78,0.78,1.6]),
(316,IRCA_BASE_PATH+"image16.nii",IRCA_BASE_PATH+"label16.nii",[0.7,0.7,1.6]),
(317,IRCA_BASE_PATH+"image17.nii",IRCA_BASE_PATH+"label17.nii",[0.74,0.74,1.6]),
(318,IRCA_BASE_PATH+"image18.nii",IRCA_BASE_PATH+"label18.nii",[0.74,0.74,2.5]),
(319,IRCA_BASE_PATH+"image19.nii",IRCA_BASE_PATH+"label19.nii",[0.7,0.7,4.])]
#(320,IRCA_BASE_PATH+"image20.nii",IRCA_BASE_PATH+"label20.nii",[0.81,0.81,2.])]

### 3 Fold
irca_test_fold1 = irca_all[:5]
irca_train_fold1 = irca_all[5:]

irca_test_fold2 = irca_all[5:10]
irca_train_fold2 = irca_all[:5] + irca_all[10:]

irca_test_fold3 = irca_all[10:15]
irca_train_fold3 = irca_all[:10]

#Datset to test
dataset = [irca_test_fold1, irca_test_fold2, irca_test_fold3]
#Paths to models. first element corresponds to first fold, second element to second fold etc.
#bs='/media/nas/03_Users/05_mohamedezz/miccai-models/'
bs='/data/miccai-models/'
models = [bs+'ID34_fold1_100k_0.9.caffemodel',bs+'ID34_fold2_12k_0.88.caffemodel',bs+'ID34_fold3_15k_0.83.caffemodel']
models_step_two = [bs+'fold1-ID29-zl60-i69k-0.53.caffemodel',bs+'fold2-ID29-zl60-i160k-0.5.caffemodel',bs+'fold3-ID29-zl60-i88k-0.48.caffemodel']

deployprototxt = [bs+'deploy.prototxt']*3
deployprototxt_step_two = deployprototxt


