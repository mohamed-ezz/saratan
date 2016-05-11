'''
Created on May 11, 2016

@author: Mohamed.Ezz

This config file configures the validation pipeline tasks

A pipeline is composed of the following components :

InputIterator 	-- iterates through input file paths. Each input is assumed to be completely independent of other inputs
Preprocessor 	-- preprocess one input
Predictor		-- runs a prediction model on an input
Postprocessor	-- postprocess predictions
Evaluator		-- evaluates prediction vs ground truth (produces metrics)
Reporter		-- aggregates and writes metrics of different inputs into some report format (e.g., CSV or Excel)

How to use the validation framework :
1- Create classes for all 6 tasks. Each of your classes inherit from one of the *Task classes in validation_task.py
	- For an example of these classes, look at saratan/validation/pipeline/example.py
2- Edit this config file so that Pipeline.InputIterator points to your inputIterator task class, and same for the other tasks
3- You task classes must implement the run method and optionally the save method
5- Run python saratan/validation/pipeline/validate.py

Notes:
1- validate.py will pass the output of each step, as an input to the next step, without really checking what it is.
	with the exception that Reporter takes also the output of InputIterator
	, it is the user's responsibility to make sure the output of one step matches the input of the next step
	, but (only) as a convention, input types should be as follows for a typical validation pipeline :
		InputIterator : no input
		Preprocessor  : takes an input tuple (input identifier, img path, ground truth path) 
		Predictor     : takes list of 2 volumes [image volume, ground truth volume]
		Postprocessor : takes list of 2 volumes [probability volume, ground truth volume]
		Evaluator     : takes a list of 2 volumes [hard label volume, ground truth volume]
		Reporter      : input tuple (from InputIterator) AND a list of metrics, e.g., [Dice, RVD, ASD]
3- You can save the intermediate results by setting [taskname]_save_to_disk=True in this config file.
	but then the task must implement the save method
4- As a convention, each task class should be defined in a separate file located in the respective directory
	under saratan/validate/pipeline/{inputiterator or preprocessor or ....}
5- The code controlling the flow of the pipeline is in saratan/validation/pipeline/validate.py
'''
from validation.pipeline.example import *

class Pipeline:
	""" Controls the flow of the pipeline """
	
	InputIterator = myInputIterator # a class that extends validation_task.InputIteratorTask
	InputIterator_save_to_disk = False
	InputIterator_save_directory= 'input_iterator_output_directory'
	
	Preprocessor  = myPreprocessor
	Preprocessor_save_to_disk = True
	Preprocessor_save_directory= 'preprocessor_output_directory'
		
	Predictor     = myPredictor
	Predictor_save_to_disk = False
	Predictor_save_directory= 'predictor_output_directory'
		
	Postprocessor = myPostprocessor
	Postprocessor_save_to_disk = True
	Postprocessor_save_directory= 'postprocessor_output_directory'	

	Evaluator     = myEvaluator
	Evaluator_save_to_disk = False
	Evaluator_save_directory= 'evluator_output_directory'
	
	Reporter      = myReporter
	Reporter_save_to_disk = True
	Reporter_save_directory = 'report_output_directory'
	








