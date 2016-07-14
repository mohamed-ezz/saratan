from validation.pipeline.validation_task import EvaluatorTask
from medpy import metric
from surface import Surface


import numpy as np


def get_scores(pred,label,vxlspacing):
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

	return volscores



class vnetEvaluator(EvaluatorTask):
	def run(self, input_tuple):


		number = input_tuple[0][0]
		vxlspacing = [1,1,1]
		pred = input_tuple[0][1]
		label = input_tuple[0][2]


		#for some reason correct label and prediction dtypes got lost
		pred = pred.astype(int)
		label = label.astype(int)

		print '#predicted pixels', np.sum(pred==1)
		print '#label pixels', np.sum(label==1)

		liver_scores = get_scores(pred>=1,label>=1,vxlspacing)
		#lesion_scores= get_scores(pred==2,label==2,vxlspacing)
		#print "Liver dice",liver_scores['dice'], "Lesion dice", lesion_scores['dice']
		#return [number,liver_scores, lesion_scores]
		print "Liver dice",liver_scores['dice']
		return [number, liver_scores]

	def save(self, directory):
		print "Saving myEvaluator to ",directory
