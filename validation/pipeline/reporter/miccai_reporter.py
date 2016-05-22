from validation.pipeline.validation_task import *
import numpy as np



class myReporter(ReporterTask):

	results = []

	def run(self, input_tuple, volumes):
		results.append([input_tuple[1][0],volumes[0],volumes[1]])

	def save_all(self, directory):
		print "Saving report to ",directory
