from validation.pipeline.validation_task import *
import numpy as np


class myInputIterator(InputIteratorTask):
	def run(self):
		print "Welcome to the MICCAI pipeline validation"
		inputs = [(1,2,3), (4,5,6), (6,7,8)]
		for i in inputs:
			yield i