from validation.pipeline.validation_task import *
import validation.pipeline.miccai_config as miccai_config
import os.path


import numpy as np



class myReporter(ReporterTask):

	#volumeid, fold, scores
	results = []

	def run(self, input_tuple, volumes):
		self.results.append([input_tuple[1][0],volumes[0],volumes[1], volumes[2]])
		print volumes

		#create line for csv file
		outstr = str(input_tuple[1][0]) + ',' + str(volumes[0]) + ','
		for l in [volumes[1],volumes[2]]:
			for k,v in l.iteritems():
				outstr += str(v) + ','
		outstr += '\n'

		#create header for csv file if necessary
		if not os.path.isfile(miccai_config.outfile):
			headerstr = 'VolumeID,Fold,'
			for k,v in volumes[1].iteritems():
				headerstr += 'Liver ' + k + ','
			for k,v in volumes[1].iteritems():
				headerstr += 'Lesion ' + k + ','

			headerstr += '\n'
			outstr = headerstr + outstr

		#write to file
		f = open(miccai_config.outfile, 'a+')
		f.write(outstr)
		f.close()

		print self.results

	def save_all(self, directory):
		print "Saving report to ",directory
