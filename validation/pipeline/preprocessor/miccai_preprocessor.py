from validation.pipeline.validation_task import *
import numpy as np



class myPreprocessor(PreprocessorTask):
	def run(self, input_tuple):
		print input_tuple
		return [np.random.random((4,4)) , np.random.random((5,5)),np.random.random((5,5))]
	def save(self, directory):
		print "Saving myPreprocessor to ",directory
