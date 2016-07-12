from validation.pipeline.validation_task import InputIteratorTask
import config as vnet_config


class vnetInputIterator(InputIteratorTask):
	def run(self):
		inputs = [(1,2,3), (4,5,6), (6,7,8), (1,2,3)]
		for i in inputs:
			yield i

	def __len__(self):
		return 1