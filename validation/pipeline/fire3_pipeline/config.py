import logging

outfile = 'output.txt'

# Image/Seg shape
slice_shape = (388,388)

skip_first_volumes = 0

###########################
##### SELECT DATASET ######
###########################


#Datset to test
dataset = ["/media/nas/sqlitedb/deepliver/niftis_parts/part4"]

fire3_bs = "/media/nas/03_Users/05_mohamedezz/thesis_latest_models/unet_models/plainunet/"
models = [fire3_bs+"fire3Best18_allslices_plainunet_wd0.001_2dropout_172.5k_0.85_0.65.caffemodel"]
deployprototxt = [fire3_bs+"deploy_plainunet.prototxt"]

