from validation.pipeline.validation_task import PreprocessorTask

import numpy as np


class vnetPreprocessor(PreprocessorTask):
	def run(self, input_tuple):
		#input tuple is [number, image, label]

		number = input_tuple[0]
		image = input_tuple[1]
		label = input_tuple[2]


		mean = np.mean(image[image>0])
		std = np.std(image[image>0])

		image -= mean
		image /= std

		print "Substraced mean and std"

		return [number, image, label]

	def save(self, directory):
		print "Saving myPreprocessor to ",directory
