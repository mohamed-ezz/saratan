import config
import logging

import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
import caffe
caffe.set_mode_gpu()


import matplotlib
from matplotlib import pyplot as plt 
matplotlib.use('Agg')
import os
from denseinference import CRFProcessor

from medpy import metric

import nibabel as nib

import numpy as np
import IPython
#this should actually be part of medpy. Apparently it isn't (anymore). So the surface.py file from http://pydoc.net/Python/MedPy/0.2.2/medpy.metric._surface/ should be manually imported
from surface import Surface

from lutils import norm_hounsfield_stat,norm_hounsfield_dyn

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8


def miccaiimshow(img,seg,preds,fname,titles=None, plot_separate_img=True):
	"""Takes raw image img, seg in range 0-2, list of predictions in range 0-2"""
	plt.figure(figsize=(25,25))
	ALPHA=1
	n_plots = len(preds)
	subplot_offset = 0

	plt.set_cmap('gray')

	if plot_separate_img:
		n_plots += 1
		subplot_offset = 1
		plt.subplot(1,n_plots,1)
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.title("Image")
		plt.axis('off')
		plt.imshow(img,cmap="gray")

	if type(preds) != list:
		preds = [preds]

	for i,pred in enumerate(preds):
		# Order of overaly
		########## OLD
		#lesion= pred==2
		#difflesion = set_minus(seg==2,lesion)
		#liver = set_minus(pred==1, [lesion, difflesion])
		#diffliver = set_minus(seg==1, [liver,lesion,difflesion])
		##########

		lesion= pred==2
		difflesion = np.logical_xor(seg==2,lesion)
		liver = pred==1
		diffliver = np.logical_xor(seg==1, liver)

		plt.subplot(1,n_plots,i+1+subplot_offset)
		title = titles[i] if titles is not None and i < len(titles) else ""
		plt.title(title)
		plt.axis('off')
		plt.imshow(img);plt.hold(True)
		# Liver prediction
		plt.imshow(np.ma.masked_where(liver==0,liver), cmap="Greens",vmin=0.1,vmax=1.2, alpha=ALPHA);plt.hold(True)
		# Liver : Pixels in ground truth, not in prediction
		plt.imshow(np.ma.masked_where(diffliver==0,diffliver), cmap="Spectral",vmin=0.1,vmax=2.2, alpha=ALPHA);plt.hold(True)

		# Lesion prediction
		plt.imshow(np.ma.masked_where(lesion==0,lesion), cmap="Blues",vmin=0.1,vmax=1.2, alpha=ALPHA);plt.hold(True)
		# Lesion : Pixels in ground truth, not in prediction
		plt.imshow(np.ma.masked_where(difflesion==0,difflesion), cmap="Reds",vmin=0.1,vmax=1.5, alpha=ALPHA)

	plt.savefig(fname)
	plt.close()
	
	


def to_scale(img, shape=None):
	if shape is None:
		shape = config.slice_shape
		
	height, width = shape
	if img.dtype == SEG_DTYPE:
		return scipy.misc.imresize(img,(height,width),interp="nearest").astype(SEG_DTYPE)
	elif img.dtype == IMG_DTYPE:
		max_ = np.max(img)
		factor = 256.0/max_ if max_ != 0 else 1
		return (scipy.misc.imresize(img,(height,width),interp="nearest")/factor).astype(IMG_DTYPE)
	else:
		raise TypeError('Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')
	
def histeq_processor(img):
	"""Histogram equalization"""
	nbr_bins=256
	#get image histogram
	imhist,bins = np.histogram(img.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum() #cumulative distribution function
	cdf = 255 * cdf / cdf[-1] #normalize
	#use linear interpolation of cdf to find new pixel values
	original_shape = img.shape
	img = np.interp(img.flatten(),bins[:-1],cdf)
	img=img/255.0
	return img.reshape(original_shape)



def downscale_img_label(imgvol,label_vol):
	"""
	Downscales an image volume and an label volume. Normalizes the hounsfield units of the image volume
	:param imgvol:
	:param label_vol:
	:return:
	"""
	imgvol = imgvol.astype(IMG_DTYPE)
	label_vol = label_vol.astype(SEG_DTYPE)
	
	imgvol_downscaled = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2]))
	label_vol_downscaled = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2]))

	# Copy image volume
	#copy_imgvol = np.copy(imgvol)
	#Truncate metal and high absorbative objects
	logging.info('Found'+str(np.sum(imgvol>1200))+'values > 1200 !!')
	imgvol[imgvol>1200] = 0
	
	for i in range(imgvol.shape[2]):
		#Get the current slc, normalize and downscale
		slc = imgvol[:,:,i]
		
		if config.ct_window_type=='dyn':
			img = norm_hounsfield_dyn(img, c_min=config.ct_window_type_min,c_max=config.ct_window_type_max)
		elif config.ct_window_type=='stat':
			img = norm_hounsfield_stat(img, c_min=config.ct_window_type_min,c_max=config.ct_window_type_max)
		else:
			print "CT Windowing did not work."

		slc = to_scale(slc, config.slice_shape)

		#slc = histeq_processor(slc)

		imgvol_downscaled[:,:,i] = slc

		#downscale the label slc for the crf
		label_vol_downscaled[:,:,i] = to_scale(label_vol[:,:,i] , config.slice_shape)
		
	return [imgvol_downscaled,label_vol_downscaled]

def scorer(pred,label,vxlspacing):
	"""

	:param pred:
	:param label:
	:param voxelspacing:
	:return:
	"""

	volscores = {}

	volscores['dice'] = metric.dc(pred,label)
	volscores['jaccard'] = metric.binary.jc(pred,label)
	volscores['voe'] = 1. - volscores['jaccard']
	volscores['rvd'] = metric.ravd(label,pred)

	if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
		volscores['assd'] = 0
		volscores['msd'] = 0
	else:
		evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
		volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
	
		volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)



	logging.info("\tDice " + str(volscores['dice']))
	logging.info("\tJaccard " + str(volscores['jaccard']))
	logging.info("\tVOE " + str(volscores['voe']))
	logging.info("\tRVD " + str(volscores['rvd']))
	logging.info("\tASSD " + str(volscores['assd']))
	logging.info("\tMSD " + str(volscores['msd']))

	return volscores

def get_average_score(scorelist,scorename,mode=None):
	"""
	:param scorelist:
	:param scorename:
	:return:
	"""

	score = 0.

	for e in scorelist:
		if mode=='abs':
			score += np.abs(e[scorename])
		else:
			score += e[scorename]

	score /= float(len(scorelist))

	return score



def zoomliver_UNET_processor(img, seg):
	""" Custom preprocessing of img,seg for UNET architecture:
	Crops the background and upsamples the found patch."""
	
	# Remove background !
	img = np.multiply(img,np.clip(seg,0,1))
	# get patch size
	col_maxes = np.max(seg, axis=0) # a row
	row_maxes = np.max(seg, axis=1)# a column
	
	nonzero_colmaxes = np.nonzero(col_maxes)[0]
	nonzero_rowmaxes = np.nonzero(row_maxes)[0]
	
	x1, x2 = nonzero_colmaxes[0], nonzero_colmaxes[-1]
	y1, y2 = nonzero_rowmaxes[0], nonzero_rowmaxes[-1]
	width = x2-x1
	height= y2-y1
	MIN_WIDTH = 60
	MIN_HEIGHT= 60
	x_pad = (MIN_WIDTH - width) / 2 if width < MIN_WIDTH else 0
	y_pad = (MIN_HEIGHT - height)/2 if height < MIN_HEIGHT else 0
	
	x1 = max(0, x1-x_pad)
	x2 = min(img.shape[1], x2+x_pad)
	y1 = max(0, y1-y_pad)
	y2 = min(img.shape[0], y2+y_pad)  
	
	img = img[y1:y2+1, x1:x2+1]
	seg = seg[y1:y2+1, x1:x2+1]
	
	img = to_scale(img, (388,388))
	seg = to_scale(seg, (388,388))
	# All non-lesion is background
	seg[seg==1]=0
	#Lesion label becomes 1
	seg[seg==2]=1
	
	# Now do padding for UNET, which takes 572x572
	#seg=np.pad(seg,((92,92),(92,92)),mode='reflect')
	img=np.pad(img,92,mode='reflect')
	return img, (x1,x2,y1,y2)


if __name__ == '__main__':
	try:
		logging.basicConfig(filename=os.path.join(config.output_dir, config.logfile), filemode='w', level=config.log_level, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
	
		#lists to calculate the overall score over all folds from, i.e. holds scores of all volumes
		overall_score_liver = []
		overall_score_lesion_crf = []
		overall_score_liver_crf = []
		overall_score_lesion = []
	
		#Iterate folds and corresponding models
		for fold, model, deployprototxt, model_step_two, deployprototxt_step_two in zip(config.dataset,config.models,config.deployprototxt, config.models_step_two, config.deployprototxt_step_two):
	
	
			logging.info("Starting new fold")
	
			#Lists to save scores for each volume of this fold
			foldscore_lesion_crf = []
			foldscore_liver_crf = []
			foldscore_liver = []
			foldscore_lesion = []
	
			#Iterate volumes in fold
			for volidx, volpaths in enumerate(fold):
	
				logging.info("Loading Network for Step 1")
				#load new network for this fold
				try:
					del net # it is a good idea to delete the net object to free up memory before instantiating another one
					net=caffe.Net(deployprototxt,model,caffe.TEST)
				except NameError:
					net=caffe.Net(deployprototxt,model,caffe.TEST)
	
	
	
				logging.info("Loading " + volpaths[1])
				imgvol = nib.load(volpaths[1]).get_data()
				labelvol = nib.load(volpaths[2]).get_data()
	
				#the raw probabilites of step 1
				probvol = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2],2))
				#the probabilites of step 2 scaled back down into the volume
				pred_step_two = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2]))
				pred_step_one = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2]))
				probvol_step_two = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2],2))
	
	
				#rotate volumes so that the networks sees them in the same orientation like during training
				imgvol = np.rot90(imgvol)
				labelvol = np.rot90(labelvol)
	
				imgvol_downscaled, labelvol_downscaled = downscale_img_label(imgvol,labelvol)
	
				#iterate slices in volume and do prediction
				logging.info("Predicting " + volpaths[1])
				for i in range(imgvol_downscaled.shape[2]):
					slc = imgvol_downscaled[:,:,i]
					#create mirrored slc for unet
					slc = np.pad(slc,((92,92),(92,92)),mode='reflect')
	
					#load slc into network and do forward pass
					net.blobs['data'].data[...] = slc
					net.forward()
	
					#now save raw probabilities
					probvol[:,:,i,:]  = net.blobs['prob'].data.transpose((0,2,3,1))[0]
					pred_step_one[:,:,i] = np.argmax(probvol[:,:,i,:], axis=2)
					#result shape is batch_img_idx , height, width, probability_of_class
	
	
				#dump probabiliteis to .npy file for future use
				#np.save('./probfiles/' + ))
				##FIX THIS
	
				logging.info("Here are the liver scores before CRF:")
				#calculate scores for liver
				pred_to_use = np.logical_or(probvol.argmax(3)==1,probvol.argmax(3)==2)
				label_to_use = np.logical_or(labelvol_downscaled==1, labelvol_downscaled==2)
	
				voxelspacing = volpaths[3]
				volumescore_liver = scorer(pred_to_use, label_to_use, voxelspacing)
	
	
				#Run Liver CRF
				logging.info("Now running CRF on Liver")
				crfparams = {'max_iterations': 10 ,'dynamic_z': True ,'ignore_memory': True ,'pos_x_std': 1.5 ,'pos_y_std': 1.5,
'pos_z_std': 1.5,'pos_w': 3.0 ,'bilateral_x_std': 9.0,'bilateral_y_std': 9.0,
'bilateral_z_std': 9.0,'bilateral_intensity_std': 20.0,'bilateral_w': 10.0}
				pro = CRFProcessor.CRF3DProcessor(**crfparams)
	
				if config.save_probability_volumes:
					np.save(os.path.join(config.output_dir , os.path.basename(volpaths[1]))+".liver.npy", probvol) 
				
				crf_pred_liver = pro.set_data_and_run(imgvol_downscaled, probvol)
	
				#calculate scores for liver
				label_to_use = np.logical_or(labelvol_downscaled==1, labelvol_downscaled==2)
	
				logging.info("Here are the liver scores after CRF:")
				volumescore_liver_crf = scorer(crf_pred_liver, label_to_use, voxelspacing)
	
				#calculate scores for lesions
				#pred_to_use = probvol.argmax(3)==2
				#label_to_use = labelvol_downscaled==2
	
				#volumescore_lesion = scorer(pred_to_use,label_to_use,voxelspacing)
	
	
	
	
				#OK, we're done on the first step of the cascaded networks and have evaluated them.
				#Now let's get to the second step.
	
				del net
				logging.info("Deleted network for cascade step 1")
				net=caffe.Net(deployprototxt_step_two,model_step_two,caffe.TEST)
	
				logging.info("Loaded network for cascade step 2")
	
				#we again iterate over all slices in the volume
				for i in range(imgvol_downscaled.shape[2]):
					slc = imgvol_downscaled[:,:,i]
					#create mirrored slc for unet
					#slc = np.pad(slc,((92,92),(92,92)),mode='reflect')
	
					#now we crop and upscale the liver
					slc_crf_pred_liver = crf_pred_liver[:, :, i].astype(SEG_DTYPE)
					#slc_crf_pred_liver = pred_to_use[:,:,i].astype(SEG_DTYPE)
					#slc_crf_pred_liver = labelvol_downscaled[:,:,i]
					if np.count_nonzero(slc_crf_pred_liver) == 0:
						probvol_step_two[:,:,i,:] = 0
					else:
						slc, bbox = zoomliver_UNET_processor(slc, slc_crf_pred_liver)
						#load slc into network and do forward pass
						net.blobs['data'].data[...] = slc
						net.forward()
		
						#scale output back down and insert into the probability volume
						
						x1,x2,y1,y2 = bbox
						leftpad, rightpad = x1, 388-x2
						toppad, bottompad = y1, 388-y2
						width, height = int(x2-x1), int(y2-y1)
						#now save probabilities
						prob = net.blobs['prob'].data.transpose((0,2,3,1))[0]
# 						probvol[:,:,i,:]  = prob

						slc_pred_step_two = np.argmax(prob,axis=2).astype(SEG_DTYPE)

						slc_pred_step_two = to_scale(slc_pred_step_two, (height,width))
						slc_pred_step_two = np.pad(slc_pred_step_two, ((toppad,bottompad),(leftpad,rightpad)), mode='constant')
						pred_step_two[:,:,i] = slc_pred_step_two

						prob0 = prob[:,:,0].astype(IMG_DTYPE) #use IMG_DTYPE bcoz we've probabiblities, not hard labels
						prob0 = to_scale(prob0, (height,width))
						prob0 = np.pad(prob0, ((toppad,bottompad),(leftpad,rightpad)), mode='constant')
# 						
# 						
						prob1 = prob[:,:,1].astype(IMG_DTYPE) 
						prob1 = to_scale(prob1, (height,width))
						prob1 = np.pad(prob1, ((toppad,bottompad),(leftpad,rightpad)), mode='constant')
						
						probvol_step_two[:,:,i,0] = prob0
						probvol_step_two[:,:,i,1] = prob1
					
					#probvol_step_two[bbox[0]:bbox[0] + bbox[1], bbox[2]:bbox[2] + bbox[3], i, :] = 
	
	
				logging.info("Lesion scores after step 2 before CRF")
				#pred_to_use = probvol_step_two.argmax(3) == 2
				pred_to_use = pred_step_two.astype(SEG_DTYPE)
				label_to_use = labelvol_downscaled == 2
	
				volumescore_lesion = scorer(pred_to_use, label_to_use, voxelspacing)
				
				# Save lesion npy probabilities
				if config.save_probability_volumes:
					np.save(os.path.join(config.output_dir , os.path.basename(volpaths[1]))+".lesion.npy", probvol_step_two) 
				
				### SAVE PLOTS
				if config.plot_every_n_slices > 0:
					for i in range(0,imgvol_downscaled.shape[2], config.plot_every_n_slices):
						pred_vol_bothsteps = pred_step_one
						pred_vol_bothsteps[pred_step_two==1] = 2
						liverdc = metric.dc(pred_step_one[:,:,i], labelvol_downscaled[:,:,i] == 1)
						lesiondc= metric.dc(pred_step_two[:,:,i], labelvol_downscaled[:,:,i] ==2)
	
						fname = os.path.join(config.output_dir , os.path.basename(volpaths[1]))
						fname += "_slc"+ str(i)+"_"
						fname += "liv"+str(liverdc)+"_les"+str(lesiondc)+".png"
						#logging.info("Plotting "+fname)
						
						miccaiimshow(imgvol_downscaled[:,:,i], labelvol_downscaled[:,:,i], [labelvol_downscaled[:,:,i],pred_vol_bothsteps[:,:,i]], fname=fname,titles=["Ground Truth","Prediction"], plot_separate_img=True)
				
			

				logging.info("Now running LESION CRF on Liver")
				crf_params = {'ignore_memory':True, 'bilateral_intensity_std': 0.16982742320252908, 'bilateral_w': 6.406401876489639, 
 						'pos_w': 2.3422381267344132, 'bilateral_x_std': 284.5377968491542, 'pos_x_std': 23.636281254341867, 
 						'max_iterations': 10}
				pro = CRFProcessor.CRF3DProcessor(**crf_params)
	
	
				crf_pred_lesion = pro.set_data_and_run(imgvol_downscaled, probvol_step_two)
				volumescore_lesion_crf = scorer(crf_pred_lesion, label_to_use, voxelspacing)
				
				
				#Append to results lists so that the average scores can be calculated later
				foldscore_liver.append(volumescore_liver)
				foldscore_lesion.append(volumescore_lesion)
				foldscore_liver_crf.append(volumescore_liver_crf)
				foldscore_lesion_crf.append(volumescore_lesion_crf)
				
				overall_score_liver_crf.append(volumescore_liver_crf)
				overall_score_lesion_crf.append(volumescore_lesion_crf)
				overall_score_liver.append(volumescore_liver)
				overall_score_lesion.append(volumescore_lesion)
	
			logging.info("=========================================")
			logging.info("Average Liver Scores before CRF for this fold: ")
			logging.info("Dice " + str(get_average_score(foldscore_liver, 'dice')))
			logging.info("Jaccard " + str(get_average_score(foldscore_liver, 'jaccard')))
			logging.info("VOE " + str(get_average_score(foldscore_liver, 'voe')))
			logging.info("RVD " + str(get_average_score(foldscore_liver, 'rvd')))
			logging.info("ASSD " + str(get_average_score(foldscore_liver, 'assd')))
			logging.info("MSD " + str(get_average_score(foldscore_liver, 'msd')))
			logging.info("=========================================")
	
			
			logging.info("=========================================")
			logging.info("Average Liver Scores after CRF for this fold: ")
			logging.info("Dice " + str(get_average_score(foldscore_liver_crf, 'dice')))
			logging.info("Jaccard " + str(get_average_score(foldscore_liver_crf, 'jaccard')))
			logging.info("VOE " + str(get_average_score(foldscore_liver_crf, 'voe')))
			logging.info("RVD " + str(get_average_score(foldscore_liver_crf, 'rvd')))
			logging.info("ASSD " + str(get_average_score(foldscore_liver_crf, 'assd')))
			logging.info("MSD " + str(get_average_score(foldscore_liver_crf, 'msd')))
			logging.info("=========================================")
	
	
			logging.info("=========================================")
			logging.info("Average Lesion Scores before CRF for this fold: ")
			logging.info("Dice " + str(get_average_score(foldscore_lesion, 'dice')))
			logging.info("Jaccard " + str(get_average_score(foldscore_lesion, 'jaccard')))
			logging.info("VOE " + str(get_average_score(foldscore_lesion, 'voe')))
			logging.info("RVD " + str(get_average_score(foldscore_lesion, 'rvd')))
			logging.info("ASSD " + str(get_average_score(foldscore_lesion, 'assd')))
			logging.info("MSD " + str(get_average_score(foldscore_lesion, 'msd')))
			logging.info("=========================================")
			
			
			logging.info("=========================================")
			logging.info("Average Lesion Scores AFTER CRF for this fold: ")
			logging.info("Dice " + str(get_average_score(foldscore_lesion_crf, 'dice')))
			logging.info("Jaccard " + str(get_average_score(foldscore_lesion_crf, 'jaccard')))
			logging.info("VOE " + str(get_average_score(foldscore_lesion_crf, 'voe')))
			logging.info("RVD " + str(get_average_score(foldscore_lesion_crf, 'rvd')))
			logging.info("ASSD " + str(get_average_score(foldscore_lesion_crf, 'assd')))
			logging.info("MSD " + str(get_average_score(foldscore_lesion_crf, 'msd')))
			logging.info("=========================================")
	
	
	
		logging.info("=========================================")
		logging.info("DONE WITH PROCESSING ALL FOLDS. NOW THE OVERALL RESULTS COME")
		logging.info("=========================================")
		logging.info("Average Liver Scores before CRF overall: ")
		logging.info("Dice " + str(get_average_score(overall_score_liver, 'dice')))
		logging.info("Jaccard " + str(get_average_score(overall_score_liver, 'jaccard')))
		logging.info("VOE " + str(get_average_score(overall_score_liver, 'voe')))
		logging.info("RVD " + str(get_average_score(overall_score_liver, 'rvd',mode='abs')))
		logging.info("ASSD " + str(get_average_score(overall_score_liver, 'assd')))
		logging.info("MSD " + str(get_average_score(overall_score_liver, 'msd')))
		logging.info("=========================================")
		logging.info("=========================================")
		logging.info("Average Liver Scores after CRF overall: ")
		logging.info("Dice " + str(get_average_score(overall_score_liver_crf, 'dice')))
		logging.info("Jaccard " + str(get_average_score(overall_score_liver_crf, 'jaccard')))
		logging.info("VOE " + str(get_average_score(overall_score_liver_crf, 'voe')))
		logging.info("RVD " + str(get_average_score(overall_score_liver_crf, 'rvd',mode='abs')))
		logging.info("ASSD " + str(get_average_score(overall_score_liver_crf, 'assd')))
		logging.info("MSD " + str(get_average_score(overall_score_liver_crf, 'msd')))
		logging.info("=========================================")
		logging.info("=========================================")
		logging.info("Average Lesion Scores before step2 CRF overall: ")
		logging.info("Dice " + str(get_average_score(overall_score_lesion, 'dice')))
		logging.info("Jaccard " + str(get_average_score(overall_score_lesion, 'jaccard')))
		logging.info("VOE " + str(get_average_score(overall_score_lesion, 'voe')))
		logging.info("RVD " + str(get_average_score(overall_score_lesion, 'rvd',mode='abs')))
		logging.info("ASSD " + str(get_average_score(overall_score_lesion, 'assd')))
		logging.info("MSD " + str(get_average_score(overall_score_lesion, 'msd')))
		logging.info("=========================================")
		logging.info("=========================================")
		logging.info("Average Lesion Scores after step2 CRF overall: ")
		logging.info("Dice " + str(get_average_score(overall_score_lesion_crf, 'dice')))
		logging.info("Jaccard " + str(get_average_score(overall_score_lesion_crf, 'jaccard')))
		logging.info("VOE " + str(get_average_score(overall_score_lesion_crf, 'voe')))
		logging.info("RVD " + str(get_average_score(overall_score_lesion_crf, 'rvd',mode='abs')))
		logging.info("ASSD " + str(get_average_score(overall_score_lesion_crf, 'assd')))
		logging.info("MSD " + str(get_average_score(overall_score_lesion_crf, 'msd')))
		logging.info("=========================================")
		
	#Creating CSV
	

		csvarray = np.zeros((len(overall_score_liver),13))
		csvarray[:,0] = range(1,len(overall_score_liver)+1)
	
		#csvarray[:,1] = [s['dice'] for s in overall_score_liver]
	
		d = overall_score_liver.iteritems()
		for i in range(6):
			d.next()[1]
			csvarray[:,i+1] = d.next()[1]
	
		d = overall_score_lesion.iteritems()
		for i in range(6):
			d.next()[1]
			csvarray[:,i+1+6] = d.next()[1]
	
	
		np.savetxt("Numbers.csv", csvarray, delimiter=",")
		
		
	except:
		logging.exception("Exception happend...")
		IPython.embed()


