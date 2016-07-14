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
2- Edit this config file so that Pipeline.InputIterator points to your inputIterator task "class", and same for the other tasks
3- Your task classes must implement the run method and optionally the save method
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
	under saratan/validate/pipeline/yourPipelineName/{inputiterator.py or preprocessor.py or....etc}
5- The code controlling the flow of the pipeline is in saratan/validation/pipeline/validate.py
'''

#for example use this
#from validation.pipeline.example import *


# from miccai_pipeline.inputiterator import myInputIterator
# from miccai_pipeline.preprocessor import myPreprocessor
# from miccai_pipeline.predictor import myPredictor
# from miccai_pipeline.postprocessor import myPostprocessor
# from miccai_pipeline.reporter import myReporter
# from miccai_pipeline.evaluator import myEvaluator

from fire3_pipeline.inputiterator import fire3InputIterator
from fire3_pipeline.preprocessor import fire3Preprocessor
from fire3_pipeline.predictor import fire3Predictor

from miccai_pipeline.inputiterator import miccaiInputIterator
from miccai_pipeline.preprocessor import miccaiPreprocessor
from miccai_pipeline.predictor import miccaiPredictor
from miccai_pipeline.postprocessor import miccaiPostprocessor
from miccai_pipeline.evaluator import miccaiEvaluator
from miccai_pipeline.reporter import miccaiReporter




from vnet_pipeline.inputiterator import vnetInputIterator
from vnet_pipeline.preprocessor import vnetPreprocessor
from vnet_pipeline.predictor import vnetPredictor
from vnet_pipeline.evaluator import vnetEvaluator
from vnet_pipeline.reporter import vnetReporter


import validation.pipeline.validation_task as validation_task
class MICCAI_Pipeline:
	""" Controls the flow of the pipeline """

	InputIterator = miccaiInputIterator # a class that extends validation_task.InputIteratorTask
	InputIterator_save_to_disk = False
	InputIterator_save_directory= 'input_iterator_output_directory'

	Preprocessor  = miccaiPreprocessor
	Preprocessor_save_to_disk = False
	Preprocessor_save_directory= 'preprocessor_output_directory'

	Predictor     = miccaiPredictor
	Predictor_save_to_disk = True
	Predictor_save_directory= 'predictor_output_directory'

	Postprocessor = miccaiPostprocessor#validation_task.IdentityPostprocessor
	Postprocessor_save_to_disk = False
	Postprocessor_save_directory= 'postprocessor_output_directory'

	Evaluator     = miccaiEvaluator#validation_task.IdentityEvaluator
	Evaluator_save_to_disk = False
	Evaluator_save_directory= 'evluator_output_directory'

	Reporter      = miccaiReporter#validation_task.IdentityReporter
	Reporter_save_directory = 'report_output_directory'

class FIRE3_Pipeline:
	""" Controls the flow of the pipeline """

	InputIterator = fire3InputIterator # a class that extends validation_task.InputIteratorTask
	InputIterator_save_to_disk = False
	InputIterator_save_directory= 'input_iterator_output_directory'

	Preprocessor  = fire3Preprocessor
	Preprocessor_save_to_disk = False
	Preprocessor_save_directory= 'preprocessor_output_directory'

	Predictor     = fire3Predictor
	Predictor_save_to_disk = True
	Predictor_save_directory= 'predictor_output_directory'

	Postprocessor = validation_task.IdentityPostprocessor
	Postprocessor_save_to_disk = False
	Postprocessor_save_directory= 'postprocessor_output_directory'

	Evaluator     = validation_task.IdentityEvaluator
	Evaluator_save_to_disk = False
	Evaluator_save_directory= 'evluator_output_directory'

	Reporter      = validation_task.IdentityReporter
	Reporter_save_directory = 'report_output_directory'

class VNET_Pipeline:
	""" Controls the flow of the pipeline """

	InputIterator = vnetInputIterator # a class that extends validation_task.InputIteratorTask
	InputIterator_save_to_disk = False
	InputIterator_save_directory= 'input_iterator_output_directory'

	Preprocessor  = vnetPreprocessor
	Preprocessor_save_to_disk = False
	Preprocessor_save_directory= 'preprocessor_output_directory'

	Predictor     = vnetPredictor
	Predictor_save_to_disk = True
	Predictor_save_directory= 'predictor_output_directory'

	Postprocessor = validation_task.IdentityPostprocessor
	Postprocessor_save_to_disk = False
	Postprocessor_save_directory= 'postprocessor_output_directory'

	Evaluator     = vnetEvaluator#validation_task.IdentityEvaluator
	Evaluator_save_to_disk = False
	Evaluator_save_directory= 'evluator_output_directory'

	Reporter      = validation_task.IdentityReporter
	Reporter_save_directory = 'report_output_directory'


###################################
####### SELECT PIPELINE ###########
###################################

Pipeline = VNET_Pipeline



