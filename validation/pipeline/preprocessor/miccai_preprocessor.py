from validation.pipeline.validation_task import *
import validation.pipeline.miccai_config as miccai_config

import numpy as np

import nibabel as nib
import scipy


IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

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


def to_scale(img, shape=None):
	if shape is None:
		shape = miccai_miccai_config.slice_shape

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

	imgvol_downscaled = np.zeros((miccai_config.slice_shape[0],miccai_config.slice_shape[1],imgvol.shape[2]))
	label_vol_downscaled = np.zeros((miccai_config.slice_shape[0],miccai_config.slice_shape[1],imgvol.shape[2]))

	# Copy image volume
	#copy_imgvol = np.copy(imgvol)
	#Truncate metal and high absorbative objects
	print 'Found'+str(np.sum(imgvol>1200))+'values > 1200 !!'
	imgvol[imgvol>1200] = 0

	for i in range(imgvol.shape[2]):
		#Get the current slc, normalize and downscale
		slc = imgvol[:,:,i]

		slc = norm_hounsfield_dyn(slc)

		slc = to_scale(slc, miccai_config.slice_shape)

		#slc = histeq_processor(slc)

		imgvol_downscaled[:,:,i] = slc

		#downscale the label slc for the crf
		label_vol_downscaled[:,:,i] = to_scale(label_vol[:,:,i] , miccai_config.slice_shape)

	return [imgvol_downscaled,label_vol_downscaled]


class myPreprocessor(PreprocessorTask):
	def run(self, input_tuple):
		print input_tuple

		#input is e.g. [1, (303, '/data/niftis_segmented/image03.nii', '/data/niftis_segmented/label03.nii', [0.62, 0.62, 1.25])]

		fold = input_tuple[0]
		voxelspacing = input_tuple[1][3]

		imgvol = nib.load(input_tuple[1][1]).get_data()
		labelvol = nib.load(input_tuple[1][2]).get_data()

		#turn 90 deg so that the network will see the images in the same orientation like during training
		imgvol = np.rot90(imgvol)
		labelvol = np.rot90(labelvol)

		imgvol_downscaled, labelvol_downscaled = downscale_img_label(imgvol,labelvol)

		return [fold,voxelspacing,imgvol_downscaled,labelvol_downscaled]

	def save(self, directory):
		print "Saving myPreprocessor to ",directory
