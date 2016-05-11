from validation.pipeline.validation_task import *
import numpy as np


class myInputIterator(InputIteratorTask):
	def run(self):
		inputs = [(1,2,3), (4,5,6), (6,7,8)]
		for i in inputs:
			yield i
	
class myPreprocessor(PreprocessorTask):
	def run(self, input_tuple):
		print input_tuple
		return [np.random.random((4,4)) , np.random.random((5,5)),np.random.random((5,5))]
	def save(self, directory):
		print "Saving myPreprocessor to ",directory

class myPredictor(PredictorTask):
	def run(self, volumes):
		print len(volumes)
		print volumes[0].shape
		print volumes[1].shape
		print volumes[2].shape
		return [np.random.random((5,5)) , np.random.random((6,6))]
	def save(self, directory):
		print "Saving myPredictor to ",directory
		
class myPostprocessor(PostprocessorTask):
	def run(self, volumes):
		print len(volumes)
		print volumes[0].shape
		print volumes[1].shape
		return [np.random.random((6,6))]
	def save(self, directory):
		print "Saving myPostprocessor to ",directory

class myEvaluator(EvaluatorTask):
	def run(self, volumes):
		print len(volumes)
		print volumes[0].shape
		return [np.random.random((6,6)) , np.random.random((7,7))]
	def save(self, directory):
		print "Saving myEvaluator to ",directory
		
class myReporter(ReporterTask):
	def run(self, input_tuple, volumes):
		print len(volumes)
		print volumes[0].shape
		print volumes[1].shape
		return [np.random.random((6,6)) , np.random.random((7,7))]
	
	def save(self, directory):
		print "Saving report to ",directory
