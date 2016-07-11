'''
Created on May 11, 2016

@author: Mohamed.Ezz

This file contains the classes that should be extended to implement validation tasks
'''
class ValidationTask(object):
	def run(self):
		""" Main function to execute this Task """
		raise NotImplementedError
	
	def save(self, directory):
		""" Saves the output of run() to disk in the given directory """ 
		raise NotImplementedError
	

class InputIteratorTask(ValidationTask):
	pass

class PreprocessorTask(ValidationTask):
	pass

class PredictorTask(ValidationTask):
	pass

class PostprocessorTask(ValidationTask):
	pass

class EvaluatorTask(ValidationTask):
	pass

class ReporterTask(ValidationTask):
	pass



# Identity tasks (they do nothing, handy to skip one of the tasks)
class IdentityPreprocessor(PreprocessorTask):
	def run(self, *args):
		return args
class IdentityPredictor(PredictorTask):
	def run(self, *args):
		return args
class IdentityPostprocessor(PostprocessorTask):
	def run(self, *args):
		return args
class IdentityReporter(ReporterTask):
	def run(self, *args):
		return args
class IdentityEvaluator(EvaluatorTask):
	def run(self, *args):
		return args
	
	
	
	