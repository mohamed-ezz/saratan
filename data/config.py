import logging

# Logging level
log_level = logging.WARNING
# Number of CPUs used for parallel processing
N_PROC = 28

# Path of created database
# This can be a list with multiple paths, but also dataset should be a list of same size
lmdb_path = ["/mnt/ID32-UNET-LiverOnlyLabel-572-liverlesion/fold1/train", "/mnt/ID32-UNET-LiverOnlyLabel-572-liverlesion/fold1/test",\
			"/mnt/ID32-UNET-LiverOnlyLabel-572-liverlesion/fold2/train", "/mnt/ID32-UNET-LiverOnlyLabel-572-liverlesion/fold2/test",\
			"/mnt/ID32-UNET-LiverOnlyLabel-572-liverlesion/fold3/train", "/mnt/ID32-UNET-LiverOnlyLabel-572-liverlesion/fold3/test"]
# Database type : lmdb or leveldb
backend = "lmdb" 
# Takes only the first n volumes. Useful to create small datasets fast
max_volumes = -1

# Image/Seg shape
slice_shape = (388,388)
# Pre-write processing
# Processors applied to images/segmentations right before persisting them to database (after augmentation...etc)
# A processor takes 2 images img and seg, and returns a tuple (img,seg)
# Available processors:
#  - processors.zoomliver_UNET_processor
#  - processors.plain_UNET_processor
#  - processors.histeq_processor
#  - processors.liveronly_label_processor
import create_ctdata as processors
processors_list = [processors.liveronly_label_processor, processors.plain_UNET_processor]

# Hounsfield Unit Windowing
# Apply static or dynamic Windowing to the CT data
# ct_window_type='dyn'
# ct_window_type_min=0.1
#ct_window_type_max=0.3

ct_window_type='stat'
ct_window_type_min=-100
ct_window_type_max=200

# Shuffle slices and their augmentations globally across the database
# You might want to set to False if dataset = test_set
shuffle_slices = True

# Augmentation factor 
augmentation_factor = 17

# whether to increase samples whose liver is small (less than 
augment_small_liver = True


# ** Labels order : tissue=0, liver=1, lesion=2
# ** We call a slice "lesion slice" if the MAX label it has is 2
# slice options: liver-lesion, stat-batch, dyn-batch
#
# liver-only:   Include only slices which are labeld with liver or lower (1 or 0)
# lesion-only:  Include only slices which are labeled with lesion or lower (2, 1 or 0)
# liver-lesion: Include only slices which are labeled with liver or lesion (slices with max=2 or with max=1)
# all slices: Include slices which are not liver or lesion with a percentage irrelevant_slice_include_prob
irrelevant_slice_include_prob=10

select_slices = "liver-lesion"

# Base path of niftis and segmentation niftis
BASE_PATH = "/media/nas/niftis_segmented"

test_set = [\
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


validation_set = [\
(97,BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/199174.nii",BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/199174_liv_x_clipped.nii"),
(10,BASE_PATH+"/CRF_048/2007-08-08/Sensation_16/segmented/05350001.nii",BASE_PATH+"/CRF_048/2007-08-08/Sensation_16/segmented/05350001_liv_4_clipped.nii"),
(3,BASE_PATH+"/CRF_035/2008-05-20/Brilliance_64/segmented/02780001.nii",BASE_PATH+"/CRF_035/2008-05-20/Brilliance_64/segmented/02780001_liv_2_clipped.nii"),
(89,BASE_PATH+"/CRF_145/2007-11-20/Sensation_40/segmented/20990001.nii",BASE_PATH+"/CRF_145/2007-11-20/Sensation_40/segmented/20990001_liv_x_clipped.nii"),
(47,BASE_PATH+"/CRF_734/2011-07-11/Biograph_6/segmented/3431744.nii",BASE_PATH+"/CRF_734/2011-07-11/Biograph_6/segmented/3431744_liv_x_clipped.nii"),
(72,BASE_PATH+"/CRF_395/2010-08-06/iCT_256/segmented/49660001.nii",BASE_PATH+"/CRF_395/2010-08-06/iCT_256/segmented/49660001_liv_x_clipped.nii"),
(26,BASE_PATH+"/CRF_174/2012-11-16/LightSpeed_Pro_32/segmented/3849520.nii",BASE_PATH+"/CRF_174/2012-11-16/LightSpeed_Pro_32/segmented/3849520_liv_x_clipped.nii"),
(98,BASE_PATH+"/CRF_335/2011-05-09/Brilliance_10/segmented/3584548.nii",BASE_PATH+"/CRF_335/2011-05-09/Brilliance_10/segmented/3584548_liv_x_clipped.nii"),
(20,BASE_PATH+"/CRF_102/2009-02-25/Aquilion/segmented/14580001.nii",BASE_PATH+"/CRF_102/2009-02-25/Aquilion/segmented/14580001_liv_4_clipped.nii"),
(34,BASE_PATH+"/CRF_223/2008-09-30/Sensation_16/segmented/3917228.nii",BASE_PATH+"/CRF_223/2008-09-30/Sensation_16/segmented/3917228_liv_x_clipped.nii"),
(29,BASE_PATH+"/CRF_174/2013-10-31/LightSpeed_Pro_32/segmented/370619.nii",BASE_PATH+"/CRF_174/2013-10-31/LightSpeed_Pro_32/segmented/370619_liv_x_clipped.nii"),
(13,BASE_PATH+"/CRF_058/2007-09-21/Sensation_40/segmented/05830001.nii",BASE_PATH+"/CRF_058/2007-09-21/Sensation_40/segmented/05830001_liv_37_clipped.nii"),
(81,BASE_PATH+"/CRF_336/2012-11-01/Birlliance_10/segmented/3557811.nii",BASE_PATH+"/CRF_336/2012-11-01/Birlliance_10/segmented/3557811_liv_x_clipped.nii"),
(39,BASE_PATH+"/CRF_710/2011-09-08/Emotion_6/segmented/84680001.nii",BASE_PATH+"/CRF_710/2011-09-08/Emotion_6/segmented/84680001_liv_1_clipped.nii"),
(65,BASE_PATH+"/CRF_502/2011-06-21/Sensation_16/segmented/65330001.nii",BASE_PATH+"/CRF_502/2011-06-21/Sensation_16/segmented/65330001_liv_x_clipped.nii"),
(96,BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/196799.nii",BASE_PATH+"/CRF_841/2014-02-12/SOMATOM_Definition_AS/segmented/196799_liv_x_clipped.nii"),
(5,BASE_PATH+"/CRF_036/2008-05-09/Emotion_Duo/segmented/03040001.nii",BASE_PATH+"/CRF_036/2008-05-09/Emotion_Duo/segmented/03040001_liv_7_clipped.nii"),
(43,BASE_PATH+"/CRF_734/2011-03-08/Sensation_16/segmented/86290001.nii",BASE_PATH+"/CRF_734/2011-03-08/Sensation_16/segmented/86290001_liv_x_clipped.nii"),
(66,BASE_PATH+"/CRF_490/2010-08-16/iCT_256/segmented/63870001.nii",BASE_PATH+"/CRF_490/2010-08-16/iCT_256/segmented/63870001_liv_x_clipped.nii"),
(92,BASE_PATH+"/CRF_118/2007-10-16/Sensation_16/segmented/17730001.nii",BASE_PATH+"/CRF_118/2007-10-16/Sensation_16/segmented/17730001_liv_x_clipped.nii")]


train_set = [\
(91,BASE_PATH+"/CRF_141/2008-01-04/Sensation_16/segmented/20500001.nii",BASE_PATH+"/CRF_141/2008-01-04/Sensation_16/segmented/20500001_liv_0_clipped.nii"),
(9,BASE_PATH+"/CRF_045/2008-12-17/LightSpeed16/segmented/04890001.nii",BASE_PATH+"/CRF_045/2008-12-17/LightSpeed16/segmented/04890001_liv_13_clipped.nii"),
(7,BASE_PATH+"/CRF_041/2009-06-17/BrightSpeed_S/segmented/04070001.nii",BASE_PATH+"/CRF_041/2009-06-17/BrightSpeed_S/segmented/04070001_liv_1_clipped.nii"),
(57,BASE_PATH+"/CRF_841/2013-08-06/SOMATOM_Definition_AS/segmented/199174.nii",BASE_PATH+"/CRF_841/2013-08-06/SOMATOM_Definition_AS/segmented/199174_liv_x_clipped.nii"),
(64,BASE_PATH+"/CRF_502/2009-09-10/Sensation_16/segmented/66020001.nii",BASE_PATH+"/CRF_502/2009-09-10/Sensation_16/segmented/66020001_liv_x_clipped.nii"),
(78,BASE_PATH+"/CRF_380/2012-09-13/LightSpeed_Pro_32/segmented/80855.nii",BASE_PATH+"/CRF_380/2012-09-13/LightSpeed_Pro_32/segmented/80855_liv_x_clipped.nii"),
(50,BASE_PATH+"/CRF_750/2011-11-10/iCT_256/segmented/86770001.nii",BASE_PATH+"/CRF_750/2011-11-10/iCT_256/segmented/86770001_liv_x_clipped.nii"),
(12,BASE_PATH+"/CRF_050/2007-11-27/Volume_Zoom/segmented/05630001.nii",BASE_PATH+"/CRF_050/2007-11-27/Volume_Zoom/segmented/05630001_liv_7_clipped.nii"),
(32,BASE_PATH+"/CRF_223/2008-04-29/Sensation_16/segmented/3911413.nii",BASE_PATH+"/CRF_223/2008-04-29/Sensation_16/segmented/3911413_liv_x_clipped.nii"),
(55,BASE_PATH+"/CRF_841/2013-01-24/SOMATOM_Definition_AS/segmented/199503.nii",BASE_PATH+"/CRF_841/2013-01-24/SOMATOM_Definition_AS/segmented/199503_liv_x_clipped.nii"),
(35,BASE_PATH+"/CRF_223/2008-10-30/Sensation_16/segmented/3916383.nii",BASE_PATH+"/CRF_223/2008-10-30/Sensation_16/segmented/3916383_liv_x_clipped.nii"),
(8,BASE_PATH+"/CRF_045/2008-09-23/LightSpeed16/segmented/04950001.nii",BASE_PATH+"/CRF_045/2008-09-23/LightSpeed16/segmented/04950001_liv_8_clipped.nii"),
(59,BASE_PATH+"/CRF_810/2012-03-06/SOMATOM_Definition_Flash/segmented/1311366.nii",BASE_PATH+"/CRF_810/2012-03-06/SOMATOM_Definition_Flash/segmented/1311366_liv_x_clipped.nii"),
(42,BASE_PATH+"/CRF_734/2011-01-17/Sensation_16/segmented/86370001.nii",BASE_PATH+"/CRF_734/2011-01-17/Sensation_16/segmented/86370001_liv_x_clipped.nii"),
(74,BASE_PATH+"/CRF_381/2012-06-19/LightSpeed_Pro_32/segmented/3615449.nii",BASE_PATH+"/CRF_381/2012-06-19/LightSpeed_Pro_32/segmented/3615449_liv_x_clipped.nii"),
(93,BASE_PATH+"/CRF_118/2008-07-24/Sensation_16/segmented/17440001.nii",BASE_PATH+"/CRF_118/2008-07-24/Sensation_16/segmented/17440001_liv_x_clipped.nii"),
(68,BASE_PATH+"/CRF_490/2011-02-09/iCT_256/segmented/63630001.nii",BASE_PATH+"/CRF_490/2011-02-09/iCT_256/segmented/63630001_liv_x_clipped.nii"),
(24,BASE_PATH+"/CRF_174/2012-08-03/LightSpeed_Pro_32/segmented/4163677.nii",BASE_PATH+"/CRF_174/2012-08-03/LightSpeed_Pro_32/segmented/4163677_liv_x_clipped.nii"),
(83,BASE_PATH+"/CRF_335/2011-03-16/Brilliance_10/segmented/3585584.nii",BASE_PATH+"/CRF_335/2011-03-16/Brilliance_10/segmented/3585584_liv_x_clipped.nii"),
(67,BASE_PATH+"/CRF_490/2010-11-19/iCT_256/segmented/63750001.nii",BASE_PATH+"/CRF_490/2010-11-19/iCT_256/segmented/63750001_liv_x_clipped.nii"),
(85,BASE_PATH+"/CRF_335/2012-01-23/Brilliance_10/segmented/3580455.nii",BASE_PATH+"/CRF_335/2012-01-23/Brilliance_10/segmented/3580455_liv_x_clipped.nii"),
(73,BASE_PATH+"/CRF_381/2012-05-10/LightSpeed_Pro_32/segmented/3617077.nii",BASE_PATH+"/CRF_381/2012-05-10/LightSpeed_Pro_32/segmented/3617077_liv_x_clipped.nii"),
(11,BASE_PATH+"/CRF_050/2007-05-11/Volume_Zoom/segmented/05680001.nii",BASE_PATH+"/CRF_050/2007-05-11/Volume_Zoom/segmented/05680001_liv_7_clipped.nii"),
(27,BASE_PATH+"/CRF_174/2013-01-25/LightSpeed_Pro_32/segmented/3849007.nii",BASE_PATH+"/CRF_174/2013-01-25/LightSpeed_Pro_32/segmented/3849007_liv_x_clipped.nii"),
(52,BASE_PATH+"/CRF_841/2012-05-03/SOMATOM_Definition_AS/segmented/204134.nii",BASE_PATH+"/CRF_841/2012-05-03/SOMATOM_Definition_AS/segmented/204134_liv_x_clipped.nii"),
(79,BASE_PATH+"/CRF_380/2014-08-06/LightSpeed_Pro_32/segmented/401545.nii",BASE_PATH+"/CRF_380/2014-08-06/LightSpeed_Pro_32/segmented/401545_liv_x_clipped.nii"),
(19,BASE_PATH+"/CRF_102/2008-11-10/Aquilion/segmented/14650001.nii",BASE_PATH+"/CRF_102/2008-11-10/Aquilion/segmented/14650001_liv_5_clipped.nii"),
(69,BASE_PATH+"/CRF_489/2011-01-26/iCT_256/segmented/62600001.nii",BASE_PATH+"/CRF_489/2011-01-26/iCT_256/segmented/62600001_liv_x_clipped.nii"),
(76,BASE_PATH+"/CRF_380/2012-04-27/LightSpeed_Pro_32/segmented/3576014.nii",BASE_PATH+"/CRF_380/2012-04-27/LightSpeed_Pro_32/segmented/3576014_liv_x_clipped.nii"),
(37,BASE_PATH+"/CRF_285/2008-05-20/Brilliance_10/segmented/3566944.nii",BASE_PATH+"/CRF_285/2008-05-20/Brilliance_10/segmented/3566944_liv_0_clipped.nii"),
(48,BASE_PATH+"/CRF_734/2011-11-11/Biograph_6/segmented/4300214.nii",BASE_PATH+"/CRF_734/2011-11-11/Biograph_6/segmented/4300214_liv_x_clipped.nii"),
(94,BASE_PATH+"/CRF_094/2007-05-15/LightSpeed_Pro_32/segmented/3731015.nii",BASE_PATH+"/CRF_094/2007-05-15/LightSpeed_Pro_32/segmented/3731015_liv_x_clipped.nii"),
(41,BASE_PATH+"/CRF_710/2011-12-02/Emotion_6/segmented/84410001.nii",BASE_PATH+"/CRF_710/2011-12-02/Emotion_6/segmented/84410001_liv_1_clipped.nii"),
(58,BASE_PATH+"/CRF_839/2012-12-10/Sensation_16/segmented/280185.nii",BASE_PATH+"/CRF_839/2012-12-10/Sensation_16/segmented/280185_liv_x_clipped.nii"),
(82,BASE_PATH+"/CRF_336/2013-02-18/Brilliance_10/segmented/3556995.nii",BASE_PATH+"/CRF_336/2013-02-18/Brilliance_10/segmented/3556995_liv_x_clipped.nii"),
(38,BASE_PATH+"/CRF_710/2011-03-28/Discovery_STE/segmented/85070001.nii",BASE_PATH+"/CRF_710/2011-03-28/Discovery_STE/segmented/85070001_liv_1_clipped.nii"),
(75,BASE_PATH+"/CRF_380/2012-03-21/LightSpeed_Pro_32/segmented/81898.nii",BASE_PATH+"/CRF_380/2012-03-21/LightSpeed_Pro_32/segmented/81898_liv_x_clipped.nii"),
(16,BASE_PATH+"/CRF_083/2008-01-17/BrightSpeed/segmented/07580001.nii",BASE_PATH+"/CRF_083/2008-01-17/BrightSpeed/segmented/07580001_liv_11_clipped.nii"),
(62,BASE_PATH+"/CRF_519/2010-09-23/iCT_256/segmented/68050001.nii",BASE_PATH+"/CRF_519/2010-09-23/iCT_256/segmented/68050001_liv_x_clipped.nii"),
(17,BASE_PATH+"/CRF_083/2009-05-19/BrightSpeed/segmented/06980001.nii",BASE_PATH+"/CRF_083/2009-05-19/BrightSpeed/segmented/06980001_liv_8_clipped.nii"),
(18,BASE_PATH+"/CRF_102/2007-12-07/Aquilion/segmented/14820001.nii",BASE_PATH+"/CRF_102/2007-12-07/Aquilion/segmented/14820001_liv_4_clipped.nii"),
(54,BASE_PATH+"/CRF_841/2012-10-08/SOMATOM_Definition_AS/segmented/201384.nii",BASE_PATH+"/CRF_841/2012-10-08/SOMATOM_Definition_AS/segmented/201384_liv_x_clipped.nii"),
(6,BASE_PATH+"/CRF_041/2009-03-18/BrightSpeed_S/segmented/04120001.nii",BASE_PATH+"/CRF_041/2009-03-18/BrightSpeed_S/segmented/04120001_liv_1_clipped.nii"),
(87,BASE_PATH+"/CRF_224/2008-09-15/Sensation_16/segmented/30280001.nii",BASE_PATH+"/CRF_224/2008-09-15/Sensation_16/segmented/30280001_liv_x_clipped.nii"),
(77,BASE_PATH+"/CRF_380/2012-07-05/LightSpeed_Pro_32/segmented/3574991.nii",BASE_PATH+"/CRF_380/2012-07-05/LightSpeed_Pro_32/segmented/3574991_liv_x_clipped.nii"),
(88,BASE_PATH+"/CRF_200/2010-11-16/iCT_256/segmented/26340001.nii",BASE_PATH+"/CRF_200/2010-11-16/iCT_256/segmented/26340001_liv_x_clipped.nii"),
(86,BASE_PATH+"/CRF_242/2007-11-27/Sensation_40/segmented/31930001.nii",BASE_PATH+"/CRF_242/2007-11-27/Sensation_40/segmented/31930001_liv_x_clipped.nii"),
(71,BASE_PATH+"/CRF_395/2010-08-06/iCT_256/segmented/49650001.nii",BASE_PATH+"/CRF_395/2010-08-06/iCT_256/segmented/49650001_liv_x_clipped.nii"),
(25,BASE_PATH+"/CRF_174/2012-09-20/LightSpeed_Pro_32/segmented/4161159.nii",BASE_PATH+"/CRF_174/2012-09-20/LightSpeed_Pro_32/segmented/4161159_liv_x_clipped.nii"),
(31,BASE_PATH+"/CRF_223/2008-02-05/Sensation_16/segmented/3920202.nii",BASE_PATH+"/CRF_223/2008-02-05/Sensation_16/segmented/3920202_liv_x_clipped.nii"),
(99,BASE_PATH+"/CRF_799/2011-10-05/Sensation_16/segmented/115160001.nii",BASE_PATH+"/CRF_799/2011-10-05/Sensation_16/segmented/115160001_liv_x_clipped.nii"),
(44,BASE_PATH+"/CRF_734/2011-04-19/Sensation_16/segmented/86200001.nii",BASE_PATH+"/CRF_734/2011-04-19/Sensation_16/segmented/86200001_liv_x_clipped.nii"),
(15,BASE_PATH+"/CRF_083/2007-12-18/BrightSpeed/segmented/07640001.nii",BASE_PATH+"/CRF_083/2007-12-18/BrightSpeed/segmented/07640001_liv_7_clipped.nii"),
(53,BASE_PATH+"/CRF_841/2012-08-15/SOMATOM_DEFINITION_AS/segmented/202539.nii",BASE_PATH+"/CRF_841/2012-08-15/SOMATOM_DEFINITION_AS/segmented/202539_liv_x_clipped.nii"),
(14,BASE_PATH+"/CRF_058/undated/Sensation_40/segmented/05860001.nii",BASE_PATH+"/CRF_058/undated/Sensation_40/segmented/05860001_liv_x_clipped.nii")]





###########################
##### 3DIRCA DATASET ######
###########################
IRCA_BASE_PATH = '/data/niftis_segmented/'
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



### 4 Fold
# irca_test_fold1 = irca_all[:5]
# irca_train_fold1 = irca_all[5:]
# irca_test_fold2 = irca_all[5:10]
# irca_train_fold2 = irca_all[:5] + irca_all[10:]
# irca_test_fold3 = irca_all[10:15]
# irca_train_fold3 = irca_all[:10] + irca_all[15:]
# irca_test_fold4 = irca_all[15:]
# irca_train_fold4 = irca_all[:15]



# Select dataset
dataset = [irca_train_fold1, irca_test_fold1,\
		irca_train_fold2, irca_test_fold2,\
		irca_train_fold3, irca_test_fold3]

