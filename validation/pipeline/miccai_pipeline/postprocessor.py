from validation.pipeline.validation_task import PostprocessorTask
import numpy as np


class miccaiPostprocessor(PostprocessorTask):
	def run(self, volumes):
		return volumes

	def save(self, directory):
		print "Saving myPostprocessor to ",directory
