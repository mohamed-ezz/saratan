'''
Created on May 11, 2016

@author: Mohamed.Ezz
'''
from config import Pipeline
import validation.pipeline.validation_task

def check_pipeline_config():
	""" Make sure pipeline config does not contain invalid settings. Not all errors can be 
	detected by this method """
	
	# Check task types
	print Pipeline.InputIterator()
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
	
	check_pipeline_config()
	
	input_iterator = Pipeline.InputIterator()
	preprocessor   = Pipeline.Preprocessor()
	predictor      = Pipeline.Predictor()
	postprocessor  = Pipeline.Postprocessor()
	evaluator      = Pipeline.Evaluator()
	reporter       = Pipeline.Reporter()
	
	for input_tuple in input_iterator.run():
		print "Input received from InputIterator"
		
		print "==== Preprocessor.run ===="
		preprocessor_output  = preprocessor.run(input_tuple)
		if Pipeline.Preprocessor_save_to_disk:
			print "==== Preprocessor.save ===="
			preprocessor.save(Pipeline.Predictor_save_directory)
			
		print "==== Predictor.run ===="
		predictor_output     = predictor.run(preprocessor_output)
		if Pipeline.Predictor_save_to_disk:
			print "==== Predictor.save ===="
			predictor.save(Pipeline.Predictor_save_directory)
		
		print "==== Postprocessor.run ===="
		postprocessor_output = postprocessor.run(predictor_output)
		if Pipeline.Postprocessor_save_to_disk:
			print "==== Postprocessor.save ===="
			postprocessor.save(Pipeline.Postprocessor_save_directory)
		
		print "==== Evaluator.run ===="
		evaluator_output     = evaluator.run(postprocessor_output)
		if Pipeline.Evaluator_save_to_disk:
			print "==== Evaluator.save ===="
			evaluator.save(Pipeline.Evaluator_save_directory)
		
		print "==== Reporter.run ===="
		# Reporter should now save the metrics in evaluator_output
		reporter_output = reporter.run(input_tuple, evaluator_output)
	
	# Reporter now saves all metrics to disk
	if Pipeline.Reporter_save_to_disk:
		print "==== Reporter.save_all ===="
		reporter.save_all(Pipeline.Reporter_save_directory)
	
	
	
	
		