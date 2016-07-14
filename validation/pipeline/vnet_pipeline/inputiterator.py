from validation.pipeline.validation_task import InputIteratorTask
import config as vnet_config

import re

import DataManager as DM
import numpy as np



class vnetInputIterator(InputIteratorTask):
	def run(self):

		dataManagerTest = DM.DataManager(vnet_config.params['ModelParams']['dirTest'], vnet_config.params['ModelParams']['dirResult'], vnet_config.params['DataManagerParams'])
		#are you serious @fausto?????
		dataManagerTest.createImageFileList()
		dataManagerTest.loadImages()
		dataManagerTest.createGTFileList()
		dataManagerTest.loadGT()

		volumes = dataManagerTest.getNumpyImages()
		labels = dataManagerTest.getNumpyGT()



		#print dataManagerTest.sitkImages

		#print dataManagerTest.sitkGT

		#print volumes

		#print labels

		#inputs = [(1,2,3), (4,5,6), (6,7,8), (1,2,3)]

		#yield (01, volumes['image01.nii'], labels['label01.nii'])
		for key in volumes:
			image_num= re.findall("\d+",key)[0]
			print key
			print image_num
			yield (image_num, volumes[key], labels['label' + str(image_num) + '.nii'])

	def __len__(self):
		return 1