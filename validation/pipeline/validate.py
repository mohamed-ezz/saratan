'''
Created on May 11, 2016

@author: Mohamed.Ezz
'''

#add relevant directory to pythonpath
import os, sys
from collections import defaultdict
sys.path.insert(1,os.path.abspath('../../'))
import logging
import time

from config import Pipeline
import validation.pipeline.validation_task

# Convenience functions for calculating and printing runtime
start_ts = defaultdict(lambda: -1)
def start_timer(timer_name="default"):
	start_ts[timer_name] = time.time()
	
def end_timer(timer_name="default"):
	assert start_ts[timer_name] != -1, "end_timer() was called without start_timer() for timer : "+timer_name
	duration = round(time.time() - start_ts[timer_name], 2)
	del start_ts[timer_name]
	print "Runtime : ", duration, "seconds"
	
def check_pipeline_config():
	""" Make sure pipeline config does not contain invalid settings. Not all errors can be 
	detected by this method """
	
	# Check task types
	assert isinstance(Pipeline.InputIterator(), validation.pipeline.validation_task.InputIteratorTask), \
		"Pipeline.InputIterator must be a subclass of validation_task.InputIteratorTask"
	assert isinstance(Pipeline.Preprocessor(), validation.pipeline.validation_task.PreprocessorTask), \
		"Pipeline.Preprocessor must be a subclass of validation_task.PreprocessorTask"
	assert isinstance(Pipeline.Predictor(), validation.pipeline.validation_task.PredictorTask), \
		"Pipeline.Predictor must be a subclass of validation_task.PredictorTask"
	assert isinstance(Pipeline.Postprocessor(), validation.pipeline.validation_task.PostprocessorTask), \
		"Pipeline.Postprocessor must be a subclass of validation_task.PostprocessorTask"
	assert isinstance(Pipeline.Evaluator(), validation.pipeline.validation_task.EvaluatorTask), \
		"Pipeline.Evaluator must be a subclass of validation_task.EvaluatorTask"
	assert isinstance(Pipeline.Reporter(), validation.pipeline.validation_task.ReporterTask), \
		"Pipeline.Reporter must be a subclass of validation_task.ReporterTask"
		
if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	# Perform some sanity checks
	check_pipeline_config()
	
	n_fails = 0
	n_succeeds = 0
	
	print "Loading modules."
	input_iterator = Pipeline.InputIterator()
	preprocessor   = Pipeline.Preprocessor()
	predictor      = Pipeline.Predictor()
	postprocessor  = Pipeline.Postprocessor()
	evaluator      = Pipeline.Evaluator()
	reporter       = Pipeline.Reporter()
	print "Modules loaded."
	
	index = 0
	total = len(input_iterator)
	for input_tuple in input_iterator.run():
		index += 1
		
		start_timer("global")
		
		try:
			progress_str = "(" + str(index)+ " / " + str(total) + ")"
			print "==========================="
			print "==========================="
			
			print "==== Preprocessor.run ====" + progress_str
			start_timer()
			preprocessor_output  = preprocessor.run(input_tuple)
			end_timer()
			if Pipeline.Preprocessor_save_to_disk:
				print "==== Preprocessor.save ====" + progress_str
				preprocessor.save(Pipeline.Predictor_save_directory)
				
			print "==== Predictor.run ====" + progress_str
			start_timer()
			predictor_output     = predictor.run(preprocessor_output)
			end_timer()
			if Pipeline.Predictor_save_to_disk:
				print "==== Predictor.save ====" + progress_str
				predictor.save(Pipeline.Predictor_save_directory)
			
			print "==== Postprocessor.run ====" + progress_str
			start_timer()
			postprocessor_output = postprocessor.run(predictor_output)
			end_timer()
			if Pipeline.Postprocessor_save_to_disk:
				print "==== Postprocessor.save ====" + progress_str
				postprocessor.save(Pipeline.Postprocessor_save_directory)
			
			print "==== Evaluator.run ====" + progress_str
			start_timer()
			evaluator_output     = evaluator.run(postprocessor_output)
			end_timer()
			if Pipeline.Evaluator_save_to_disk:
				print "==== Evaluator.save ====" + progress_str
				evaluator.save(Pipeline.Evaluator_save_directory)
			
			print "==== Reporter.run ====" + progress_str
			# Reporter should now save the metrics in evaluator_output
			start_timer()
			reporter_output = reporter.run(input_tuple, evaluator_output)
			end_timer()
			
			end_timer("global")
				
			n_succeeds += 1
		except:
			n_fails += 1
			logging.exception("Failed to process input : " + str(input_tuple))

	n_total = n_fails + n_succeeds
	logging.info("Validation is done : %i / %i failed." % (n_fails, n_total))
	
	
	
	
		