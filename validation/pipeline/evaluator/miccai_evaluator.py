from validation.pipeline.validation_task import *
import numpy as np


class myEvaluator(EvaluatorTask):
	def run(self, volumes):
		print len(volumes)
		print volumes[0].shape
		return [np.random.random((6,6)) , np.random.random((7,7))]
	def save(self, directory):
		print "Saving myEvaluator to ",directory
