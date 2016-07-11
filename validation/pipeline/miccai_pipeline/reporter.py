from validation.pipeline.validation_task import ReporterTask
import config as miccai_config
import os.path
import numpy as np



class miccaiReporter(ReporterTask):

	def __init__(self):
		self.results = []
		self.outpath = os.path.join(miccai_config.outdir, miccai_config.outfile)

	def run(self, input_tuple, foldidx_scores):
		foldidx, liver_scores, lesion_scores = foldidx_scores
		volume_id = input_tuple[1][0]
		self.results.append([volume_id, foldidx, liver_scores, lesion_scores])

		#create line for csv file
		outstr = str(volume_id) + ',' + str(foldidx) + ','
		for l in [liver_scores, lesion_scores]:
			for k,v in l.iteritems():
				outstr += str(v) + ','
		outstr += '\n'

		#create header for csv file if necessary
		if not os.path.isfile(self.outpath):
			headerstr = 'VolumeID,Fold,'
			for k,v in liver_scores.iteritems():
				headerstr += 'Liver_' + k + ','
			for k,v in liver_scores.iteritems():
				headerstr += 'Lesion_' + k + ','

			headerstr += '\n'
			outstr = headerstr + outstr

		#write to file
		f = open(self.outpath, 'a+')
		f.write(outstr)
		f.close()

