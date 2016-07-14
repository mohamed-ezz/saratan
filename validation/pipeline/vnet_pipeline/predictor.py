from validation.pipeline.validation_task import PredictorTask
import config as vnet_config

import matplotlib.pyplot as plt

import os
import numpy as np

import caffe
caffe.set_mode_gpu()


class vnetPredictor(PredictorTask):


	def run(self, input_tuple):

		number = input_tuple[0]
		image = input_tuple[1]
		label = input_tuple[2]

		print 'image shape',image.shape
		print 'image unique', np.unique(image)

		print 'label unique', np.unique(label)

		plt.imshow(label[:,:,30])


		net = caffe.Net(vnet_config.params['ModelParams']['prototxtTest'],
                        os.path.join(vnet_config.params['ModelParams']['dirSnapshots'],"_iter_" + str(vnet_config.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

		btch = np.reshape(image,[1,1,image.shape[0],image.shape[1],image.shape[2]])
		net.blobs['data'].data[...] = btch
		out = net.forward()
		#print out
		l = out["labelmap"]

		#print np.unique(l)
		#labelmap = np.squeeze(l[0,1,:,:,:])
		#prediction = np.squeeze(labelmap)
		prediction= np.argmax(l[0],axis=0)

		#plt.imshow(l[0,0, :,:,30])
		plt.savefig('img.jpg')


		print 'prediction np unique', np.unique(prediction)


		return [number, prediction, label]

	def save(self, directory):
		print "Saving myPredictor to ",directory