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



class myEvaluator(EvaluatorTask):
	def run(self, volumes):

		fold = volumes[0]
		vxlspacing = volumes[1]
		pred = volumes[2]
		label = volumes[3]

		#for some reason correct label and prediction dtypes got lost
		pred = pred.astype(int)
		label = label.astype(int)


		return [fold,get_scores(pred>=1,label>=1,vxlspacing),get_scores(pred==2,label==2,vxlspacing)]

	def save(self, directory):
		print "Saving myEvaluator to ",directory
