from validation.pipeline.validation_task import *
import numpy as np


class myPostprocessor(PostprocessorTask):
	def run(self, volumes):
		return volumes

	def save(self, directory):
		print "Saving myPostprocessor to ",directory
