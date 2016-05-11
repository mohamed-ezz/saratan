#! /usr/bin/env python

import nlopt
import numpy as np

import logging
import config

from denseinference import CRFProcessor

from multiprocessing import Pool, Manager

import nibabel as nib

import scipy.misc
import os


import medpy.metric

#global list for volumes
volumes = []

#best results so far
best_dice = -1
best_params = None

n_iterations = 0

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

def to_scale(img, shape=None):
	if shape is None:
		shape = config.slice_shape
		
	height, width = shape
	if img.dtype == SEG_DTYPE:
		return scipy.misc.imresize(img,(height,width),interp="nearest").astype(SEG_DTYPE)
	elif img.dtype == IMG_DTYPE:
		factor = 256.0/np.max(img)
		return (scipy.misc.imresize(img,(height,width),interp="nearest")/factor).astype(IMG_DTYPE)
	else:
		raise TypeError('Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')
	
	
def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
	""" Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
	# calc min and max
	min,max = np.amin(arr), np.amax(arr)
	if min <= 0:
		arr = np.clip(arr, min * c_min, max * c_max)
		# right shift to zero
		arr = np.abs(min * c_min) + arr
	else:
		arr = np.clip(arr, min, max * c_max)
		# left shift to zero
		arr = arr - min
	# normalization
	norm_fac = np.amax(arr)
	if norm_fac != 0:
		norm = np.divide(
				np.multiply(arr,255),
			 	np.amax(arr))
	else:  # don't divide through 0
		norm = np.multiply(arr, 255)
		
	norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
	return norm


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
	img=img/256.0
	return img.reshape(original_shape)


def process_img_label(imgvol,segvol):
	"""
	Process a given image volume and its label and return arrays as a new copy
	:param imgvol:
	:param label_vol:
	:return:
	"""
	imgvol_downscaled = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2]))
	segvol_downscaled = np.zeros((config.slice_shape[0],config.slice_shape[1],imgvol.shape[2]))
	imgvol[imgvol>1200] = 0

	for i in range(imgvol.shape[2]):
		#Get the current slice, normalize and downscale
		slice = np.copy(imgvol[:,:,i])
		slice = norm_hounsfield_dyn(slice)
		slice = to_scale(slice, config.slice_shape)
		slice = histeq_processor(slice)
		imgvol_downscaled[:,:,i] = slice
		#downscale the label slice for the crf
		segvol_downscaled[:,:,i] = to_scale(segvol[:,:,i], config.slice_shape)
		
	return [imgvol_downscaled, segvol_downscaled]



def crf_worker(img,label,probvol,crfsettings):
	"""
	Worker function for Parallel CRF Processing of multiple Volumes
	:param img:
	:param label:
	:param prob:
	:param crfsettings:
	:return:  dice
	"""
	pro = CRFProcessor.CRF3DProcessor(**crfsettings)
	#print "started crf"
	#print np.min(img), np.max(img)
	result = pro.set_data_and_run(img, probvol)
	#print np.unique(result)
	#print "done with crf"
	_dice = medpy.metric.dc(result==1,label==config.target_label)
	print "Dice of single volume: " + str(_dice)

	#not sure if that's necessary
	del pro
	
	return _dice




def run_crf(params, grad):
	"""
	:param pos_x_std:
	:param bilateral_x_std:
	:param bilateral_intensity_std:
	:param pos_w:
	:param bilateral_w:
	:return:
	"""
	global best_dice, best_params, volumes, n_iterations

	n_iterations += 1
	#Stupid NLopt it always wants a grad even for algorithms that don't use gradient. If grad is not empty, something is wrong.
	#print grad

	pos_x_std , pos_y_std , pos_z_std, bilateral_x_std, bilateral_y_std, bilateral_z_std, bilateral_intensity_std, pos_w, bilateral_w = params

	
# 	logging.info("=======================")
# 	logging.info("Running CRF with the following parameters:")
# 	logging.info("pos x std: " + str(pos_x_std))
# 	logging.info("pos y std: " + str(pos_y_std))
# 	logging.info("pos z std: " + str(pos_z_std))
# 	logging.info("pos w: " + str(pos_w))
# 	logging.info("bilateral x std: " + str(bilateral_x_std))
# 	logging.info("bilateral y std: " + str(bilateral_y_std))
# 	logging.info("bilateral z std: " + str(bilateral_z_std))
# 	logging.info("bilateral intensity std: " + str(bilateral_intensity_std))
# 	logging.info("bilateral w: " + str(bilateral_w))

	#Here's something to come
	crfsettings = dict(max_iterations=config.max_iterations,
                                    pos_x_std=pos_x_std,
                                    pos_y_std=pos_y_std,
                                    pos_z_std=pos_z_std,
                                    pos_w=pos_w,
                                    bilateral_x_std=bilateral_x_std,
                                    bilateral_y_std=bilateral_y_std,
                                    bilateral_z_std=bilateral_z_std,
                                    bilateral_intensity_std=bilateral_intensity_std,
                                    bilateral_w=bilateral_w,

                                    dynamic_z=config.dynamic_z,
                                    ignore_memory=config.ignore_memory)


	#list of dice scores
	dices = []
	#list of pipes
	results = []

	pool = Pool(processes=config.N_PROC)

		
	#start results
	for img, label, voxelsize, prob in volumes:
		# Normalize z std according to volume's voxel slice spacing
		copy_crfsettings = dict(crfsettings)
		copy_crfsettings['pos_z_std'] *= voxelsize[2] # z std grows with larger spacing between slices
		results.append(pool.apply_async(crf_worker,(img,label,prob,crfsettings)))
		#dices.append(crf_worker(img,label,prob,crfsettings))

	#get results
	for p in results:
		dices.append(p.get())

	pool.close()

	dice_average = np.average(dices)

	logging.info("-----------------------")
	logging.info("Iteration : " + str(n_iterations))
	logging.info("Best avg dice was: " + str(best_dice))
	logging.info("   with best params : "+ str(best_params))
	logging.info("Current avg dice is: " + str(dice_average))
	logging.info("   with current params :" + str(params))
	logging.info("=======================")

	if dice_average >= best_dice:
		best_params = params
		best_dice = dice_average
		print 'FOUND BETTER PARAMS'
	

	return dice_average






if __name__ == '__main__':
	logging.basicConfig(filename=config.logfile, level=config.log_level, format='%(levelname)s:%(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')

	#also print logs to stdout (ok, stderr technically...)
	logging.getLogger().addHandler(logging.StreamHandler())

	logging.info("Preparing volumes")

	dices_b4_crf = []
	for vol_id, img_path, seg_path, voxelsize, prob_path in config.dataset:
		imgvol = nib.load(os.path.normpath(img_path)).get_data()
		segvol = nib.load(os.path.normpath(seg_path)).get_data()
		probvol = np.load(prob_path)

		#rotate volumes so that the networks sees them in the same orientation like during training
		imgvol = np.rot90(imgvol)
		segvol = np.rot90(segvol)

		imgvol, segvol = process_img_label(imgvol,segvol)

		volumes.append([imgvol,segvol,voxelsize, probvol])
		
		_dice = medpy.metric.dc(probvol.argmax(3)==1,segvol==config.target_label)
		dices_b4_crf.append(_dice)
		
		print "Dice before CRF: " + str(_dice)
	
	print "Average dice before CRF :", np.mean(dices_b4_crf)

	logging.info("Setting up Optimiser")
	#create optimiser object. We have 9 free parameters.
	opt = nlopt.opt(nlopt.LN_BOBYQA, 9)
	opt.set_lower_bounds([0.000000001]*8+[1]) #fix bilateral_weight to 1
	opt.set_upper_bounds([550]*8+[1])

	#The optimizer is supposed to maximise the dice that is returned by run_crf
	opt.set_max_objective(run_crf)
	opt.set_stopval(.99)
	#opt.set_maxtime(config.MAX_N_IT)

	#Runs optimization
	logging.info("Running Optimisation")
	paramsopt = opt.optimize(config.params_initial)

	logging.info(str(paramsopt))
	logging.info("Done")
	
	
	
	
	
	
	