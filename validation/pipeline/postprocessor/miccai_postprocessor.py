from validation.pipeline.validation_task import *
import numpy as np

class myPostprocessor(PostprocessorTask):
	def run(self, volumes):
		print len(volumes)
		print volumes[0].shape
		print volumes[1].shape
		return [np.random.random((6,6))]
	def save(self, directory):
		print "Saving myPostprocessor to ",directory
