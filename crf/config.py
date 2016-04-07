import logging

# Logging level
log_level = logging.INFO

logfile = 'output.txt'

# Number of CPUs used for parallel processing
N_PROC = 14

#Maximum number of iterations before optimisation is stopped
MAX_N_IT = -1


# Image/Seg shape
slice_shape = (388,388)

#Initial Parameters
params_initial = [\
1.5,  # pos_x_std
1.5,  # pos_y_std
1.5,  # pos_z_std
9.0,  # bilateral_x_std
9.0,  # bilateral_y_std
9.0,  # bilateral_z_std
20.0, # bilateral_intensity_std
3.0,  # pos_w
1.0   # bilateral_w  #we fix this one during optimization
]

#Fixed CRF Parameters
max_iterations = 20
dynamic_z = False
ignore_memory = True




###########################
##### 3DIRCA DATASET ######
###########################

#Use this line on IBBM computers
IRCA_BASE_PATH = '/media/nas/01_Datasets/CT/Abdomen/3Dircadb1/niftis_segmented_lesions/'
#Use the Following line for slu02
#IRCA_BASE_PATH = '/data/niftis_segmented/'

#the array after the label element is the voxel spacing
IRCA_PROBS_PATH = "/media/nas/03_Users/05_mohamedezz/prob_volumes/"
IRCA_PROBS_SUFFIX = ".lesion.npy"
irca_all= [\
(301,IRCA_BASE_PATH+"image01.nii",IRCA_BASE_PATH+"label01.nii",[0.57,0.57,1.6],IRCA_PROBS_PATH+"image01.nii"+IRCA_PROBS_SUFFIX),
(302,IRCA_BASE_PATH+"image02.nii",IRCA_BASE_PATH+"label02.nii",[0.78,0.78,1.6],IRCA_PROBS_PATH+"image02.nii"+IRCA_PROBS_SUFFIX),
(303,IRCA_BASE_PATH+"image03.nii",IRCA_BASE_PATH+"label03.nii",[0.62,0.62,1.25],IRCA_PROBS_PATH+"image03.nii"+IRCA_PROBS_SUFFIX),
(304,IRCA_BASE_PATH+"image04.nii",IRCA_BASE_PATH+"label04.nii",[0.74,0.74,2.],IRCA_PROBS_PATH+"image04.nii"+IRCA_PROBS_SUFFIX),
#(305,IRCA_BASE_PATH+"image05.nii",IRCA_BASE_PATH+"label05.nii",[0.78,0.78,1.6],IRCA_PROBS_PATH+"image05.nii"+IRCA_PROBS_SUFFIX),
(306,IRCA_BASE_PATH+"image06.nii",IRCA_BASE_PATH+"label06.nii",[0.78,0.78,1.6],IRCA_PROBS_PATH+"image06.nii"+IRCA_PROBS_SUFFIX),
#(307,IRCA_BASE_PATH+"image07.nii",IRCA_BASE_PATH+"label07.nii",[0.78,0.78,1.6],IRCA_PROBS_PATH+"image07.nii"+IRCA_PROBS_SUFFIX),
(308,IRCA_BASE_PATH+"image08.nii",IRCA_BASE_PATH+"label08.nii",[0.56,0.56,1.6],IRCA_PROBS_PATH+"image08.nii"+IRCA_PROBS_SUFFIX),
(309,IRCA_BASE_PATH+"image09.nii",IRCA_BASE_PATH+"label09.nii",[0.87,0.87,2.],IRCA_PROBS_PATH+"image09.nii"+IRCA_PROBS_SUFFIX),
(310,IRCA_BASE_PATH+"image10.nii",IRCA_BASE_PATH+"label10.nii",[0.73,0.73,1.6],IRCA_PROBS_PATH+"image10.nii"+IRCA_PROBS_SUFFIX),
#(311,IRCA_BASE_PATH+"image11.nii",IRCA_BASE_PATH+"label11.nii",[0.72,0.72,1.6],IRCA_PROBS_PATH+"image11.nii"+IRCA_PROBS_SUFFIX),
(312,IRCA_BASE_PATH+"image12.nii",IRCA_BASE_PATH+"label12.nii",[0.68,0.68,1.],IRCA_PROBS_PATH+"image12.nii"+IRCA_PROBS_SUFFIX),
(313,IRCA_BASE_PATH+"image13.nii",IRCA_BASE_PATH+"label13.nii",[0.67,0.67,1.6],IRCA_PROBS_PATH+"image13.nii"+IRCA_PROBS_SUFFIX),
#(314,IRCA_BASE_PATH+"image14.nii",IRCA_BASE_PATH+"label14.nii",[0.72,0.72,1.6],IRCA_PROBS_PATH+"image14.nii"+IRCA_PROBS_SUFFIX),
(315,IRCA_BASE_PATH+"image15.nii",IRCA_BASE_PATH+"label15.nii",[0.78,0.78,1.6],IRCA_PROBS_PATH+"image15.nii"+IRCA_PROBS_SUFFIX),
(316,IRCA_BASE_PATH+"image16.nii",IRCA_BASE_PATH+"label16.nii",[0.7,0.7,1.6],IRCA_PROBS_PATH+"image016.nii"+IRCA_PROBS_SUFFIX),
(317,IRCA_BASE_PATH+"image17.nii",IRCA_BASE_PATH+"label17.nii",[0.74,0.74,1.6],IRCA_PROBS_PATH+"image17.nii"+IRCA_PROBS_SUFFIX),
(318,IRCA_BASE_PATH+"image18.nii",IRCA_BASE_PATH+"label18.nii",[0.74,0.74,2.5],IRCA_PROBS_PATH+"image18.nii"+IRCA_PROBS_SUFFIX),
(319,IRCA_BASE_PATH+"image19.nii",IRCA_BASE_PATH+"label19.nii",[0.7,0.7,4.],IRCA_PROBS_PATH+"image19.nii"+IRCA_PROBS_SUFFIX)]
#(320,IRCA_BASE_PATH+"image20.nii",IRCA_BASE_PATH+"label20.nii",[0.81,0.81,2.],IRCA_PROBS_PATH+"image20.nii"+IRCA_PROBS_SUFFIX)]

irca_test_fold1 = irca_all[:5]
irca_train_fold1 = irca_all[5:]

irca_test_fold2 = irca_all[5:10]
irca_train_fold2 = irca_all[:5] + irca_all[10:]

irca_test_fold3 = irca_all[10:15]
irca_train_fold3 = irca_all[:10] + irca_all[15:]

irca_test_fold4 = irca_all[15:]
irca_train_fold4 = irca_all[:15]



# Select dataset
#dataset = [irca_train_fold1, irca_test_fold1,\
#		irca_train_fold2, irca_test_fold2,\
#		irca_train_fold3, irca_test_fold3,\
#		irca_train_fold4, irca_test_fold4]
#


#Datset to test
dataset = irca_all[:1]
