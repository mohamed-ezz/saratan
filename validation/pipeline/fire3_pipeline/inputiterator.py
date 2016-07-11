from validation.pipeline.validation_task import InputIteratorTask
import config as fire3_config
import numpy as np
import os

class fire3InputIterator(InputIteratorTask):
	def run(self):
		print "Welcome to the MICCAI pipeline validation"

		file_index = 1
		skips = 0
		for filename in fire3_config.dataset:
			if skips < fire3_config.skip_first_volumes:
				skips += 1
				continue
			
			assert os.path.exists(filename), "InputIterator could not find file: "+filename
			f = open(filename, 'r')
			lines = f.readlines()
			
			for line in lines:
				line = line.strip() #remove \n
				
				if line.endswith('.gz'):
					line = line[:-3]
				
				yield [file_index, line]
			
			file_index += 1
			
	def __len__(self):
		fold_lens = map(lambda filename: len(open(filename,'r').readlines()), fire3_config.dataset)
		return sum(fold_lens)