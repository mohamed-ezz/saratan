import nlopt
import numpy as np

import logging
import config

from caffehelper import nifti_helper as nh
from denseinference import CRFProcessor

from multiprocessing import Pool

import nibabel as nib

import scipy as sp

import os


import medpy.metric

#global list for volumes
volumes = []

#best results so far
best_dice = -1
best_params = None

def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
	""" Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
	# calc min and max
	min,max = np.amin(arr), np.amax(arr)
	arr = arr.astype(IMG_DTYPE)
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

def process_img_label(imgvol,segvol):
	"""
	Process a given image volume and its label 
	:param imgvol:
	:param label_vol:
	:return:
	"""
	
	imgvol[imgvol>1200] = 0
	imgvol = norm_hounsfield_dyn(imgvol)

	return [imgvol, segvol]



def crf_worker(img,label,prob,crfsettings):
	"""
	Worker function for Parallel CRF Processing of multiple Volumes
	:param img:
	:param label:
	:param prob:
	:param crfsettings:
	:return: lesion dice
	"""
	pro = CRFProcessor.CRF3DProcessor(**crfsettings)
	print "started crf"
	result = pro.set_data_and_run(img, probvol)
	print np.unique(result)
	print "done with crf"
	lesion_dice = medpy.metric.dc(result==1,label==2)
	print "Lesion dice of volume: " + str(lesion_dice)

	#not sure if that's necessary
	del pro
	
	return lesion_dice




def run_crf(params, grad):
	"""

	:param pos_x_std:
	:param bilateral_x_std:
	:param bilateral_intensity_std:
	:param pos_w:
	:param bilateral_w:
	:return:
	"""

	#Stupid NLopt it always wants a grad even for algorithms that don't use gradient. If grad is not empty, something is wrong.
	print grad

	pos_x_std , pos_y_std , pos_z_std, bilateral_x_std, bilateral_y_std, bilateral_z_std, bilateral_intensity_std, pos_w, bilateral_w = params

	
	logging.info("=======================")
	logging.info("Running CRF with the following parameters:")
	logging.info("pos x std: " + str(pos_x_std))
	logging.info("pos y std: " + str(pos_y_std))
	logging.info("pos z std: " + str(pos_z_std))
	logging.info("pos w: " + str(pos_w))
	logging.info("bilateral x std: " + str(bilateral_x_std))
	logging.info("bilateral y std: " + str(bilateral_y_std))
	logging.info("bilateral z std: " + str(bilateral_z_std))
	logging.info("bilateral intensity std: " + str(bilateral_intensity_std))
	logging.info("bilateral w: " + str(bilateral_w))

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
	processes = []

	pool = Pool(processes=config.N_PROC)

	#start processes
	for img, label, prob in volumes:
		processes.append(pool.apply_async(crf_worker,(img,label,prob,crfsettings)))

	#get results
	for p in processes:
		print "waiting for result"
		dices.append(p.get())
		print "received result"

	pool.close()

	dice_average = np.average(dices)


	if dice_average >= best_dice:
		best_params = params
		best_dice = dice_average
		
	
	logging.info("-----------------------")
	logging.info("Current avg lesion dice result: " + str(dice_average))
	logging.info("   with current params :" + str(params))
	logging.info("Best avg dice so far was: " + str(best_dice))
	logging.info("   with best params : "+ str(best_params))
	logging.info("=======================")

	return dice_average






if __name__ == '__main__':
	logging.basicConfig(filename=config.logfile, level=config.log_level, format='%(levelname)s:%(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')

	#also print logs to stdout (ok, stderr technically...)
	logging.getLogger().addHandler(logging.StreamHandler())

	logging.info("Preparing volumes")

	for img_path, seg_path, voxelsize, prob_path in config.dataset:
		imgvol = nib.load(os.path.normpath(img_path)).get_data()
		segvol = nib.load(os.path.normpath(seg_path)).get_data()
		probvol = np.load(prob_path)

		#rotate volumes so that the networks sees them in the same orientation like during training
		imgvol = np.rot90(imgvol)
		segvol = np.rot90(segvol)

		imgvol, segvol = process_img_label(imgvol,segvol)

		volumes.append([imgvol,segvol,probvol])

	print np.unique(segvol)
	print "Dice before CRF: " + str(medpy.metric.dc(probvol.argmax(3)==1,segvol==2))

	logging.info("Setting up Optimiser")
	#create optimiser object. We have 9 free parameters.
	opt = nlopt.opt(nlopt.LN_BOBYQA, 9)
	opt.set_lower_bound([0.000000001]*8+[1]) #fix bilateral_weight to 1
	opt.set_upper_bound([550]*8+[1])

	#The optimizer is supposed to maximise the dice that is returned by run_crf
	opt.set_max_objective(run_crf)
	opt.set_stopval(.99)
	#opt.set_maxtime(config.MAX_N_IT)

	#Runs optimization
	logging.info("Running Optimisation")
	paramsopt = opt.optimize(config.params_initial)

	print paramsopt
	logging.info(str(paramsopt))
	logging.info("Done")
	
	
	
	
	
	
	