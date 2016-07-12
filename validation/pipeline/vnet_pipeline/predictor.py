from validation.pipeline.validation_task import PredictorTask
import config as vnet_config

import os
import numpy as np

import caffe

class vnetPredictor(PredictorTask):


	def run(self, volumes):

		net = caffe.Net(vnet_config.params['ModelParams']['prototxtTest'],
                        os.path.join(vnet_config.params['ModelParams']['dirSnapshots'],"_iter_" + str(vnet_config.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

		return [np.random.random((5,5)) , np.random.random((6,6))]
	def save(self, directory):
		print "Saving myPredictor to ",directory