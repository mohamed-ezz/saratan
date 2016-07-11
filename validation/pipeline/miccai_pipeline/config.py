import logging

# Number of CPUs used for parallel processing
#N_PROC = 14

outdir = "/data/thesis_results/fire3_cfcn_crf_47"
outfile = 'fire3_output_cfcn_crf.txt'
# Image/Seg shape
slice_shape = (388,388)

#################
#### OUTPUT #####
#################

# Logging level
#log_level = logging.INFO
#output_dir = "."
#logfile = 'output.txt'
# Save liver.npy and lesion.npy volumes to output_dir/[niftiname].liver.npy, with shape h,w,slices,classes
#save_probability_volumes = True
# Save slices as png files. This param is the increment between plotting one slice and the next
# Set 0 or -1 to disable plotting
#plot_every_n_slices = -1

###########################
##### FIRE3 DATASET ######
###########################

apply_histeq = False
apply_liver_crf = True
use_net2 = True #for CFCN
net1_n_classes = 2

FIRE3_BASE_PATH = "/media/nas/niftis_segmented"

fire3_validation_set = [\
(3,FIRE3_BASE_PATH+"/CRF_035/2008-05-20/Brilliance_64/segmented/02780001.nii",FIRE3_BASE_PATH+"/CRF_035/2008-05-20/Brilliance_64/segmented/02780001_liv_2_clipped.nii"),
(10,FIRE3_BASE_PATH+"/CRF_048/2007-08-08/Sensation_16/segmented/05350001.nii",FIRE3_BASE_PATH+"/CRF_048/2007-08-08/Sensation_16/segmented/05350001_liv_4_clipped.nii"),
(97,FIRE3_BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/199174.nii",FIRE3_BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/199174_liv_x_clipped.nii"),
(89,FIRE3_BASE_PATH+"/CRF_145/2007-11-20/Sensation_40/segmented/20990001.nii",FIRE3_BASE_PATH+"/CRF_145/2007-11-20/Sensation_40/segmented/20990001_liv_x_clipped.nii"),
(47,FIRE3_BASE_PATH+"/CRF_734/2011-07-11/Biograph_6/segmented/3431744.nii",FIRE3_BASE_PATH+"/CRF_734/2011-07-11/Biograph_6/segmented/3431744_liv_x_clipped.nii"),
(72,FIRE3_BASE_PATH+"/CRF_395/2010-08-06/iCT_256/segmented/49660001.nii",FIRE3_BASE_PATH+"/CRF_395/2010-08-06/iCT_256/segmented/49660001_liv_x_clipped.nii"),
(26,FIRE3_BASE_PATH+"/CRF_174/2012-11-16/LightSpeed_Pro_32/segmented/3849520.nii",FIRE3_BASE_PATH+"/CRF_174/2012-11-16/LightSpeed_Pro_32/segmented/3849520_liv_x_clipped.nii"),
(98,FIRE3_BASE_PATH+"/CRF_335/2011-05-09/Brilliance_10/segmented/3584548.nii",FIRE3_BASE_PATH+"/CRF_335/2011-05-09/Brilliance_10/segmented/3584548_liv_x_clipped.nii"),
(20,FIRE3_BASE_PATH+"/CRF_102/2009-02-25/Aquilion/segmented/14580001.nii",FIRE3_BASE_PATH+"/CRF_102/2009-02-25/Aquilion/segmented/14580001_liv_4_clipped.nii"),
(34,FIRE3_BASE_PATH+"/CRF_223/2008-09-30/Sensation_16/segmented/3917228.nii",FIRE3_BASE_PATH+"/CRF_223/2008-09-30/Sensation_16/segmented/3917228_liv_x_clipped.nii"),
(29,FIRE3_BASE_PATH+"/CRF_174/2013-10-31/LightSpeed_Pro_32/segmented/370619.nii",FIRE3_BASE_PATH+"/CRF_174/2013-10-31/LightSpeed_Pro_32/segmented/370619_liv_x_clipped.nii"),
(13,FIRE3_BASE_PATH+"/CRF_058/2007-09-21/Sensation_40/segmented/05830001.nii",FIRE3_BASE_PATH+"/CRF_058/2007-09-21/Sensation_40/segmented/05830001_liv_37_clipped.nii"),
(81,FIRE3_BASE_PATH+"/CRF_336/2012-11-01/Birlliance_10/segmented/3557811.nii",FIRE3_BASE_PATH+"/CRF_336/2012-11-01/Birlliance_10/segmented/3557811_liv_x_clipped.nii"),
(39,FIRE3_BASE_PATH+"/CRF_710/2011-09-08/Emotion_6/segmented/84680001.nii",FIRE3_BASE_PATH+"/CRF_710/2011-09-08/Emotion_6/segmented/84680001_liv_1_clipped.nii"),
(65,FIRE3_BASE_PATH+"/CRF_502/2011-06-21/Sensation_16/segmented/65330001.nii",FIRE3_BASE_PATH+"/CRF_502/2011-06-21/Sensation_16/segmented/65330001_liv_x_clipped.nii"),
(96,FIRE3_BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/196799.nii",FIRE3_BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/196799_liv_x_clipped.nii"),
(5,FIRE3_BASE_PATH+"/CRF_036/2008-05-09/Emotion_Duo/segmented/03040001.nii",FIRE3_BASE_PATH+"/CRF_036/2008-05-09/Emotion_Duo/segmented/03040001_liv_7_clipped.nii"),
(43,FIRE3_BASE_PATH+"/CRF_734/2011-03-08/Sensation_16/segmented/86290001.nii",FIRE3_BASE_PATH+"/CRF_734/2011-03-08/Sensation_16/segmented/86290001_liv_x_clipped.nii"),
#(66,FIRE3_BASE_PATH+"/CRF_490/2010-08-16/iCT_256/segmented/63870001.nii",FIRE3_BASE_PATH+"/CRF_490/2010-08-16/iCT_256/segmented/63870001_liv_x_clipped.nii"),
(92,FIRE3_BASE_PATH+"/CRF_118/2007-10-16/Sensation_16/segmented/17730001.nii",FIRE3_BASE_PATH+"/CRF_118/2007-10-16/Sensation_16/segmented/17730001_liv_x_clipped.nii")]

fire3_dataset = [fire3_validation_set[4:5]]
fire3_bs = "/media/nas/03_Users/05_mohamedezz/thesis_latest_models/unet_models/"
fire3_models = [fire3_bs+"step1_models/fire3_overfit_allslices_200k_0.89.caffemodel"]
fire3_models_step_two = [fire3_bs+"step2_models/fire3_step2_wd0.0005_2dropout_191.5k_0.6.caffemodel"]
fire3_deployprototxt = [fire3_bs+"step1_models/deploy_step1.prototxt"]
fire3_deployprototxt_step_two = [fire3_bs+"step2_models/deploy_step2.prototxt"]

fire3 = [fire3_dataset, fire3_models, fire3_deployprototxt, fire3_models_step_two, fire3_deployprototxt_step_two]
###########################
##### 3DIRCA DATASET ######
###########################

#Use this line on IBBM computers
#IRCA_BASE_PATH = '/media/nas/01_Datasets/CT/Abdomen/3Dircadb1/niftis_segmented_lesions/'
#Use the Following line for slu02
IRCA_BASE_PATH = '/media/nas/01_Datasets/CT/Abdomen/3Dircadb1/niftis_segmented_lesions/'

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

#irca_dataset = [irca_test_fold1, irca_test_fold2, irca_test_fold3]
irca_dataset = [irca_all]
irca_bs='/media/nas/03_Users/05_mohamedezz/thesis_latest_models/unet_models/'
irca_models = [irca_bs+'step1_models/fire3_overfit_allslices_200k_0.89.caffemodel']
irca_models_step_two = [irca_bs+"step2_models/fire3_step2_wd0.0005_2dropout_191.5k_0.6.caffemodel"]
irca_deployprototxt = 			[irca_bs+'step1_models/deploy_step1.prototxt']
irca_deployprototxt_step_two = 	[irca_bs+'step1_models/deploy_step1.prototxt']

irca = [irca_dataset, irca_models, irca_deployprototxt, irca_models_step_two, irca_deployprototxt_step_two]
###########################
##### SELECT DATASET ######
###########################

dataset, models, deployprototxt, models_step_two, deployprototxt_step_two = fire3







