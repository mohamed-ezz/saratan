from validation.pipeline.validation_task import *
import validation.pipeline.miccai_config as miccai_config
import scipy
import caffe
caffe.set_mode_gpu()

from denseinference import CRFProcessor


import numpy as np



IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

def to_scale(img, shape=None):
	if shape is None:
		shape = miccai_config.slice_shape

	height, width = shape
	if img.dtype == SEG_DTYPE:
		return scipy.misc.imresize(img,(height,width),interp="nearest").astype(SEG_DTYPE)
	elif img.dtype == IMG_DTYPE:
		max_ = np.max(img)
		factor = 256.0/max_ if max_ != 0 else 1
		return (scipy.misc.imresize(img,(height,width),interp="nearest")/factor).astype(IMG_DTYPE)
	else:
		raise TypeError('Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')


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



class myPredictor(PredictorTask):


	def __init__(self, fold=1):
		self.net=caffe.Net(miccai_config.deployprototxt[fold-1],miccai_config.models[fold-1],caffe.TEST)
		self.fold = fold
		self.cascade = 1

	def updatenet(self,fold,cascade):
		if fold != self.fold or cascade != self.cascade:
			print "Updating network"
			if cascade == 1:
				netparams = miccai_config.deployprototxt[fold-1],miccai_config.models[fold-1],caffe.TEST
			elif cascade == 2:
				netparams = miccai_config.deployprototxt_step_two[fold-1],miccai_config.models_step_two[fold-1],caffe.TEST
			elif True:
				print "Are you sane??"

			try:
				del self.net
				self.net=caffe.Net(*netparams)
				self.fold = fold
				self.cascade = cascade
			except NameError:
				self.net=caffe.Net(*netparams)
				self.fold = fold
				self.cascade = cascade


	def run(self, volumes):

		
		fold = volumes[0]
		voxelspacing = volumes[1]
		imgvol_downscaled = volumes[2]
		labelvol_downscaled = volumes[3]


		#the raw probabilites of step 1
		probvol = np.zeros((miccai_config.slice_shape[0],miccai_config.slice_shape[1],imgvol_downscaled.shape[2],2))
		#the probabilites of step 2 scaled back down into the volume
		pred_step_two = np.zeros((miccai_config.slice_shape[0],miccai_config.slice_shape[1],imgvol_downscaled.shape[2]))
		probvol_step_two = np.zeros((miccai_config.slice_shape[0],miccai_config.slice_shape[1],imgvol_downscaled.shape[2],2))


		print "Running Step 1"

		self.updatenet(fold=fold,cascade=1)

		for i in range(imgvol_downscaled.shape[2]):
			slc = imgvol_downscaled[:,:,i]
			#create mirrored slc for unet
			slc = np.pad(slc,((92,92),(92,92)),mode='reflect')

			#load slc into network and do forward pass
			self.net.blobs['data'].data[...] = slc
			self.net.forward()

			#now save raw probabilities
			probvol[:,:,i,:]  = self.net.blobs['prob'].data.transpose((0,2,3,1))[0]

			#result shape is batch_img_idx , height, width, probability_of_class
			
		
		#return [fold,voxelspacing, np.argmax(probvol, axis=3),labelvol_downscaled]
	
	
	
	
		print "Running CRF"

		crfparams = {'max_iterations': 10 ,'dynamic_z': True ,'ignore_memory': True ,'pos_x_std': 1.5 ,'pos_y_std': 1.5,
'pos_z_std': 1.5,'pos_w': 3.0 ,'bilateral_x_std': 9.0,'bilateral_y_std': 9.0,
'bilateral_z_std': 9.0,'bilateral_intensity_std': 20.0,'bilateral_w': 10.0}
		pro = CRFProcessor.CRF3DProcessor(**crfparams)

		crf_pred = pro.set_data_and_run(imgvol_downscaled, probvol)


		#Now let's get to the second step.

		print "Running Step 2"
		self.updatenet(fold=fold,cascade=2)

		#we again iterate over all slices in the volume
		for i in range(imgvol_downscaled.shape[2]):
			slc = imgvol_downscaled[:,:,i]

			#now we crop and upscale the liver
			slc_crf_pred_liver = crf_pred[:, :, i].astype(SEG_DTYPE)

			if np.count_nonzero(slc_crf_pred_liver) == 0:
				probvol_step_two[:,:,i,:] = 0
			else:
				slc, bbox = zoomliver_UNET_processor(slc, slc_crf_pred_liver)
				#load slc into network and do forward pass
				self.net.blobs['data'].data[...] = slc
				self.net.forward()

				#scale output back down and insert into the probability volume

				x1,x2,y1,y2 = bbox
				leftpad, rightpad = x1, 388-x2
				toppad, bottompad = y1, 388-y2
				width, height = int(x2-x1), int(y2-y1)
				#now save probabilities
				prob = self.net.blobs['prob'].data.transpose((0,2,3,1))[0]
# 						probvol[:,:,i,:]  = prob

				slc_pred_step_two = np.argmax(prob,axis=2).astype(SEG_DTYPE)

				slc_pred_step_two = to_scale(slc_pred_step_two, (height,width))
				slc_pred_step_two = np.pad(slc_pred_step_two, ((toppad,bottompad),(leftpad,rightpad)), mode='constant')
				pred_step_two[:,:,i] = slc_pred_step_two


		#merge pred_ste_two into the prediction from step 1

		crf_pred[pred_step_two==1] = 2


		return [fold,voxelspacing,crf_pred,labelvol_downscaled]






	def save(self, directory):
		print "Saving myPredictor to ",directory
