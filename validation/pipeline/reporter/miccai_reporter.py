from validation.pipeline.validation_task import *
import numpy as np



class myReporter(ReporterTask):
	def run(self, input_tuple, volumes):
		print len(volumes)
		print volumes[0].shape
		print volumes[1].shape
		return [np.random.random((6,6)) , np.random.random((7,7))]

	def save_all(self, directory):
		print "Saving report to ",directory
