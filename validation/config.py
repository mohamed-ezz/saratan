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
##### FIRE3  DATASET ######
###########################

# Base path of niftis and segmentation niftis
BASE_PATH = "/media/nas/niftis_segmented"


fire3_test_set = [\
(2,BASE_PATH+"/CRF_035/2008-05-20/Brilliance_64/segmented/02770001.nii",BASE_PATH+"/CRF_035/2008-05-20/Brilliance_64/segmented/02770001_liv_1_clipped.nii"),
(4,BASE_PATH+"/CRF_036/2008-02-04/Emotion_Duo/segmented/03150001.nii",BASE_PATH+"/CRF_036/2008-02-04/Emotion_Duo/segmented/03150001_liv_3_clipped.nii"),
(21,BASE_PATH+"/CRF_103/2008-11-14/Brilliance_6_Dunlee/segmented/3774867.nii",BASE_PATH+"/CRF_103/2008-11-14/Brilliance_6_Dunlee/segmented/3774867_liv_x_clipped.nii"),
(23,BASE_PATH+"/CRF_143/2007-12-04/LightSpeed_Pro_32/segmented/3816280.nii",BASE_PATH+"/CRF_143/2007-12-04/LightSpeed_Pro_32/segmented/3816280_liver_0_clipped.nii"),
(22,BASE_PATH+"/CRF_103/2009-07-29/Brilliance_6_Dunlee/segmented/3772968.nii",BASE_PATH+"/CRF_103/2009-07-29/Brilliance_6_Dunlee/segmented/3772968_liv_x_clipped.nii"),
(33,BASE_PATH+"/CRF_223/2008-07-22/Sensation_16/segmented/3917859.nii",BASE_PATH+"/CRF_223/2008-07-22/Sensation_16/segmented/3917859_liv_x_clipped.nii"),
(36,BASE_PATH+"/CRF_223/2009-03-27/Sensation_16/segmented/28840001.nii",BASE_PATH+"/CRF_223/2009-03-27/Sensation_16/segmented/28840001_liv_x_clipped.nii"),
(28,BASE_PATH+"/CRF_174/2013-04-12/LightSpeed_Pro_32/segmented/372109.nii",BASE_PATH+"/CRF_174/2013-04-12/LightSpeed_Pro_32/segmented/372109_liv_x_clipped.nii"),
(45,BASE_PATH+"/CRF_734/2011-06-28/Sensation_16/segmented/86110001.nii",BASE_PATH+"/CRF_734/2011-06-28/Sensation_16/segmented/86110001_liv_x_clipped.nii"),
(46,BASE_PATH+"/CRF_734/2011-04-19/undefined/segmented/86180001.nii",BASE_PATH+"/CRF_734/2011-04-19/undefined/segmented/86180001_liv_x_clipped.nii"),
(49,BASE_PATH+"/CRF_750/2011-09-23/iCT_256/segmented/86910001.nii",BASE_PATH+"/CRF_750/2011-09-23/iCT_256/segmented/86910001_liv_x_clipped.nii"),
(30,BASE_PATH+"/CRF_174/2013-12-06/LightSpeed_Pro_32/segmented/368984.nii",BASE_PATH+"/CRF_174/2013-12-06/LightSpeed_Pro_32/segmented/368984_liv_x_clipped.nii"),
(56,BASE_PATH+"/CRF_841/2013-01-24/SOMATOM_Definition_AS/segmented/200424.nii",BASE_PATH+"/CRF_841/2013-01-24/SOMATOM_Definition_AS/segmented/200424_liv_x_clipped.nii"),
(61,BASE_PATH+"/CRF_756/2011-06-17/Definition_AS+/segmented/87840001.nii",BASE_PATH+"/CRF_756/2011-06-17/Definition_AS+/segmented/87840001_liv_x_clipped.nii"),
(63,BASE_PATH+"/CRF_512/2011-10-18/SOMATOM_Definition_AS/segmented/66090001.nii",BASE_PATH+"/CRF_512/2011-10-18/SOMATOM_Definition_AS/segmented/66090001_liv_x_clipped.nii"),
(70,BASE_PATH+"/CRF_485/2010-08-11/Emotion_16_(2007)/segmented/3292245.nii",BASE_PATH+"/CRF_485/2010-08-11/Emotion_16_(2007)/segmented/3292245_liv_x_clipped.nii"),
(80,BASE_PATH+"/CRF_336/2012-04-26/Brilliance_10/segmented/3558746.nii",BASE_PATH+"/CRF_336/2012-04-26/Brilliance_10/segmented/3558746_liv_x_clipped.nii"),
(84,BASE_PATH+"/CRF_335/2011-10-21/Brilliance_10/segmented/3581986.nii",BASE_PATH+"/CRF_335/2011-10-21/Brilliance_10/segmented/3581986_liv_x_clipped.nii"),
(90,BASE_PATH+"/CRF_141/2007-11-22/Sensation_16/segmented/20610001.nii",BASE_PATH+"/CRF_141/2007-11-22/Sensation_16/segmented/20610001_liv_x_clipped.nii"),
(100,BASE_PATH+"/CRF_272/2008-10-30/Defenition/segmented/34180001.nii",BASE_PATH+"/CRF_272/2008-10-30/Defenition/segmented/34180001_liv_x_clipped.nii")]



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
dataset = [fire3_test_set]
#Paths to models. first element corresponds to first fold, second element to second fold etc.
#bs='/media/nas/03_Users/05_mohamedezz/miccai-models/'
bs='/data/miccai-models/'
models = [bs+'ID34_fold1_100k_0.9.caffemodel',bs+'ID34_fold2_12k_0.88.caffemodel',bs+'ID34_fold3_15k_0.83.caffemodel']
models_step_two = [bs+'fold1-ID29-zl60-i69k-0.53.caffemodel',bs+'fold2-ID29-zl60-i160k-0.5.caffemodel',bs+'fold3-ID29-zl60-i88k-0.48.caffemodel']

deployprototxt = [bs+'deploy.prototxt']*3
deployprototxt_step_two = deployprototxt


