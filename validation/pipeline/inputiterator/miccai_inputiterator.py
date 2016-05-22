from validation.pipeline.validation_task import *
import validation.pipeline.miccai_config as miccai_config
import numpy as np


class myInputIterator(InputIteratorTask):
	def run(self):
		print "Welcome to the MICCAI pipeline validation"

		fold_index = 1
		for fold in miccai_config.dataset:
			for input in fold:
				yield [fold_index, input]
			fold_index += 1