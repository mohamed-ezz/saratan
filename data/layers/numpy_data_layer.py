'''
Created on Apr 6, 2016

@author: Mohamed.Ezz

This module includes Caffe python data layers to read volumes directly from Npy files (3D CT volumes).

The layer scales well with large amounts of data, and supports prefetching for minimal processing overhead.

'''
import sys, os, time, random, shutil
import numpy as np
import lmdb, caffe, nibabel
from multiprocessing import Pool, Process, Queue
from Queue import Empty, Full
import scipy.misc, scipy.ndimage.interpolation
from tqdm import tqdm 
import plyvel
from itertools import izip
import logging
from contextlib import closing


## Deformation Augmentation
from skimage.transform import PiecewiseAffineTransform, warp

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

#Prefetching queue
MAX_QUEUE_SIZE = 1000
PREFETCH_BATCH_SIZE = 100


def maybe_true(probability=0.5):
	rnd = random.random()
	return rnd <= probability

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
		#norm = (arr*255)/ norm_fac
		norm = np.divide(
				np.multiply(arr,255),
			 	np.amax(arr))
	else:  # don't divide through 0
		norm = np.multiply(arr, 255)
		
	norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
	return norm


class augmentation:
	
### Core functions
	@staticmethod
	def _get_shift(img, seg, x, y):
		"""Move pixel in a direction by attaching on the other side. (i.e. x=5 -> 5 pixel to the right; y=-7 seven pixel down)
		:param id: slice id in current volume
		:return: Shifted img and seg"""
		# slide in x direction
		if x != 0:
			img = np.append(img[x:,:], img[:x,:], axis=0)
			seg = np.append(seg[x:,:], seg[:x,:], axis=0)
		# slide in y direction
		if y != 0:
			img = np.append(img[:,-y:], img[:,:-y], axis=1)
			seg = np.append(seg[:,-y:], seg[:,:-y], axis=1)
		return img, seg

	@staticmethod
	def _crop(img, seg, crop_type, frac=0.95):
	
		height, width = img.shape
		
		if crop_type == 'lt':
			box = (0                       , 0,
				   int(round(width * frac)), int(round(height * frac)))
		elif crop_type == 'rt':
			box = (int(round((1.0 - frac) * width)), 0,
				   width                           , int(round(height * frac)))
		elif crop_type == 'lb':
			box = (0                       , int(round((1.0 - frac) * height)),
				   int(round(width * frac)), height)
		elif crop_type == 'rb':
			box = (int(round((1.0 - frac) * width)),int(round((1.0 - frac) * height)),
				   width,                           height)
		elif crop_type == 'c':
			box = (int(round((1.0 - frac) * (width/2.0))),       int(round((1.0 - frac) * (height/2.0))),
				   int(round(width * (frac + (1 - frac) / 2.0))),int(round(height * (frac + (1 - frac) / 2.0))))
		else:
			raise ValueError("Wrong crop_type. Must be lt, rt, lb, rb or c.")
		# Do the cropping
		x1, y1, x2, y2 = box
		img, seg = img[y1:y2, x1:x2], seg[y1:y2, x1:x2]

		return img, seg
	
	@staticmethod
	def _rotate(img, angle):
		# Prevent augmentation with no rotation, otherwise the same image will be appended
		if angle==0:
			angle=1
		# rotate without interpolation (order=0 makes it take nearest pixel)
		rotated = scipy.ndimage.interpolation.rotate(img, angle, order=0)
		#rotation results in extra pixels on the borders
		# We fix it assuming square shape
		assert img.shape[0] == img.shape[1], "Given image for rotation is not of square shape :" + str(img.shape)
		extra = rotated.shape[0]-img.shape[0]
		extra_left = extra/2
		extra_right = extra - extra_left
		rotated = rotated[extra_left: -extra_right, extra_left: - extra_right]
		return rotated
	
#####################################
######    PUBLIC FUNCTIONS    #######
#####################################

	@staticmethod
	def identity(img,seg):
		""" return original slices...."""
		return img,seg
	@staticmethod
	def noise(img, seg):
		img_noisy = (img + 0.3 * img.std() * np.random.random(img.shape)).astype(IMG_DTYPE)
		return img_noisy, seg
	@staticmethod
	def get_shift_up(img, seg):
		height = img.shape[0]
		return augmentation._get_shift(img,seg, 0, int(height/15))
	@staticmethod
	def get_shift_down(img, seg):
		height = img.shape[0]
		return augmentation._get_shift(img,seg, 0, -int(height/15))
	@staticmethod
	def get_shift_left(img, seg):
		width = img.shape[1]
		return augmentation._get_shift(img,seg, -int(width/15),   0)
	@staticmethod
	def get_shift_right(img, seg):
		width = img.shape[1]
		return augmentation._get_shift(img,seg,  int(width/15),   0)
	@staticmethod
	def crop_lb(img, seg):
		return augmentation._crop(img,seg, 'lb')
	@staticmethod
	def crop_rt(img, seg):
		return augmentation._crop(img,seg,'rt')
	@staticmethod
	def crop_c(img, seg):
		return augmentation._crop(img,seg,'c')
	@staticmethod
	def rotate(img, seg):
		rand = random.randrange(-10,10)
		return augmentation._rotate(img, rand), augmentation._rotate(seg, rand) 
			
	

class processors:

	@staticmethod
	def histeq_processor(img, seg):
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
		return img.reshape(original_shape), seg
	
	@staticmethod
	def plain_UNET_processor(img,seg):
		img = to_scale(img, (388,388))
		seg = to_scale(seg, (388,388))
		# Now do padding for UNET, which takes 572x572
		#seg=np.pad(seg,((92,92),(92,92)),mode='reflect')
		img=np.pad(img,92,mode='reflect')
		return img, seg
	
	@staticmethod
	def liveronly_label_processor(img, seg):
		"""Converts lesion labels to liver label. The resulting classifier classifies liver vs. background."""
		seg[seg==2]=1
		return img,seg
	
	@staticmethod
	def remove_non_liver(img, seg):
		# Remove background !
		img = np.multiply(img,np.clip(seg,0,1))
		return img, seg
	
	@staticmethod
	def zoomliver_UNET_processor(img, seg):
		""" Custom preprocessing of img,seg for UNET architecture:
		Crops the background and upsamples the found patch."""
		
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
		x_pad = int((MIN_WIDTH - width) / 2.0 if width < MIN_WIDTH else 0)
		y_pad = int((MIN_HEIGHT - height)/2.0 if height < MIN_HEIGHT else 0)
		
		# Additional padding to make sure boundary lesions are included
		#SAFETY_PAD = 15
		#x_pad += SAFETY_PAD
		#y_pad += SAFETY_PAD
		
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
		return img, seg
	


import config


class NumpyDataLayer(caffe.Layer):
	""" Caffe Data layer that reads directly from npy files """
	
	
	def setup(self, bottom, top):
		print "Setup NumpyDataLayer"
		self.top_names = ['data', 'label']
		
		self.batch_size = 1 #current batch_size>1 is not implemented. but very simple to implement in the forward() function
		self.img_volumes = [] # list of numpy volumes
		self.seg_volumes = [] # list of numpy label volumes
		
		self.n_volumes= 0 # number of volumes in dataset
		self.n_augmentations = config.augmentation_factor #number of possible augmentations
		self.queue = Queue(MAX_QUEUE_SIZE)
		self.n_total_slices = 0		
		for vol_id, img_path, seg_path in self.dataset :
			# shape initially is like 512,512,129
			imgvol = np.load(img_path,mmap_mode='r')
			imgvol = np.rot90(imgvol) #rotate so that spine is down, not left
			imgvol = np.transpose(imgvol, (2,0,1)) # bring slice index to first place
			self.img_volumes.append(imgvol)
			
			segvol = np.load(seg_path,mmap_mode='r')
			segvol = np.rot90(segvol)
			segvol = np.transpose(segvol, (2,0,1))
			self.seg_volumes.append(segvol)
			
			assert imgvol.shape == segvol.shape, "Volume and segmentation have different shapes: %s vs. %s" % (str(imgvol.shape),str(segvol.shape))
			
			self.n_volumes+= 1
			self.n_total_slices += segvol.shape[0]
		
		print "Dataset has ", self.n_total_slices,"(before augmentation)"
		top[0].reshape(1,1,572,572)
		top[1].reshape(1,1,388,388)
		
		# Seed the random generator
		np.random.seed(123)
		# Put first input into queue
		child_seed = np.random.randint(0,9000)
		# The child_seed is a randomly generated seed and it is needed because
		# without it, every newly created process will be identical and will generate
		# the same sequence of random numbers
		self.p = Process(target = self.prepare_next_batch, args=(child_seed,))
		self.p.start()
		
		import atexit
		def cleanup():
			print "Terminating dangling process"
			self.p.terminate()
			self.p.join()
		atexit.register(cleanup)
		
		import signal
		signal.signal(signal.SIGINT, cleanup)
		
	def reshape(self, bottom, top):
		pass
	
	def forward(self, bottom, top):
		while True:
			try:
				img, seg = self.queue.get(timeout=1)
				break
			except Empty: #If queue is empty for any reason, must get_next_slice now
				# Make sure that there is no self.p currently running 
				if not self.p.is_alive():
					# be 100% sure to terminate self.p
					self.p.join()
					print "forward(): Queue was empty. Spawing prefetcher and retrying"
					child_seed = np.random.randint(0,9000)
					self.p = Process(target = self.prepare_next_batch, args=(child_seed,))
					self.p.start()
		
		top[0].data[0,...] = img
		top[1].data[0,...] = seg
		
		#self.p.join()
		
		#child_seed = np.random.randint(0,9000)
		#self.pool_result = self.ppool.apply_async(self, args=(child_seed,))
		#self.pool_result.get()
		#self.p = Process(target = self.prepare_next_batch, args=(child_seed,))
		#self.p.start()
		
	def backward(self, top, propagate_down, bottom):
		pass

		
	def prepare_next_batch(self, seed):
		np.random.seed(seed)
		for _ in range(PREFETCH_BATCH_SIZE):
			self.get_next_slice()
		
	
	def get_next_slice(self):
		""" Randomly pick a next slice and push it to the shared queue """
		while True:
			# Pick random slice and augmentation
			# Doing it this way, each volume has equal probability of being selected regardless of 
			# how many slices it has.
			# Each slice inside the volume has equal chances to be picked.
			# But globally, not every slice has the same probablity of being selected,
			# it depends on how many other slices in its same volume is competing with it.
			vol_idx   = np.random.randint(0,self.n_volumes) 
			slice_idx = np.random.randint(0,self.img_volumes[vol_idx].shape[0])
			aug_idx   = np.random.randint(0,self.n_augmentations)
			
			img = self.img_volumes[vol_idx][slice_idx] 
			seg = self.seg_volumes[vol_idx][slice_idx]
			
			#print vol_idx, slice_idx, aug_idx
			# Only break if we found a relevant slice
			if self.is_relevant_slice(seg):
				break
		
		img, seg = self.prepare_slice(img, seg, aug_idx)
		
		try:
			self.queue.put((img, seg))
		except Full:
			pass
		
	def is_relevant_slice(self, slc):
		""" Checks whether a given segmentation slice is relevant, according to rule specified in config.select_slices (e.g., lesion-only)"""
		
		# We increase small livers by rejecting non-small liver slices more frequently
		if config.more_small_livers:
			n_liver = 1.0 * np.sum(slc > 0)
			if (100*n_liver/slc.size) > config.small_liver_percent: # NOT small liver
				return maybe_true(0.7)
				
				
		
		if config.select_slices == "all":
			# Reject half of the slices that has no liver/lesion
			if np.count_nonzero(slc) == 0:
				return maybe_true(0.3)
			return True
			
		
		max = np.max(slc)
		if config.select_slices == "liver-lesion":
			return max == 1 or max == 2
		elif config.select_slices == "lesion-only":
			return max == 2
		elif config.select_slices == "liver-only":
			return max == 1
		else:
			raise ValueError("Invalid value for config.select_slices :", config.select_slices)

	def prepare_slice(self, img, seg, aug_idx):
		# Make sure 0 >= label >= 2
		seg = np.clip(seg, 0, 2)
		img = norm_hounsfield_dyn(img)
		img, seg = self.augment_slice(img, seg, aug_idx)
		for processor in config.processors_list:
			img, seg = processor(img, seg)
		#img = to_scale(img, (400,400))
        #seg = to_scale(seg, (400,400))
		return img, seg

	def augment_slice(self, img, seg, aug_idx):
		
		aug_func = [augmentation.identity,
					augmentation.crop_lb,
					augmentation.crop_rt,
					augmentation.crop_c,
					augmentation.rotate,
					augmentation.rotate,
					augmentation.get_shift_up,
					augmentation.get_shift_down,
					augmentation.get_shift_left,
					augmentation.get_shift_right]
					#augmentation.noise

		#Invoke the selected augmentation function
		img, seg = aug_func[aug_idx](img, seg)		
		return img, seg
	
class NumpyTrainDataLayer(NumpyDataLayer):
	""" NumpyDataLayer for the Train dataset """
	def setup(self, bottom, top):
		self.dataset = config.train_dataset
		super(NumpyTrainDataLayer, self).setup(bottom, top)

class NumpyTestDataLayer(NumpyDataLayer):
	""" NumpyDataLayer for the Test dataset """
	def setup(self, bottom, top):
		self.dataset = config.test_dataset
		super(NumpyTestDataLayer, self).setup(bottom, top)
	
	
