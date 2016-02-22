'''
Contains common functions for reading data out of leveldb

@author: Mohamed.Ezz
'''
import plyvel
import numpy as np
from caffe.proto import caffe_pb2

def denormalize_img_255(arr):
	""" Denormalizes a nparray to 0-255 values """
	min = arr.min()
	max = arr.max()

	new = (arr - min) * (255.0/(max-min))
	return new.astype(np.uint8)

def leveldb_arrays(leveldbdir):
	""" Generator. Given leveldb directory, iterate the stored data as numpy arrays. Yields (Key, NumpyArray) """
	db = plyvel.DB(leveldbdir)
	for k,v in db.iterator():
		yield k, to_numpy_matrix(v)

def nth_datum(leveldb, n):
	""" Returns nth datum. 0-based index"""
	n+=1
	it = leveldb.iterator()
	for _ in range(n):
		_, v = it.next()
	datum = caffe_pb2.Datum()
	datum.ParseFromString(v)
	return datum

def get_data_type(datum):
	""" By simple calculations, conclude the size of integers stored in datum.data """
	n_values = datum.height * datum.width * datum.channels
	n_bytes  = len(datum.data)
	int_size = float(n_bytes) / n_values
	if int_size != int(int_size) or int_size not in [1,2,4,8]:
		raise ValueError("Can't find int size. n_values : %i , n_bytes : %i" % (n_values, n_bytes))
	types = {1:np.int8, 2:np.int16, 4:np.int32, 8:np.int64}
	type_ = types[int(int_size)]
	return type_

def find_keycount(leveldb, count_values=None):
	""" Takes a plyvel.DB instance and returns number of keys found and count of each value. 
	count_values is a list of values to count, e.g. count_values=[0,1,2] will return [count of 1s, count of 2s, count of 3s]
	if count_values is None, return value of this function is [],key_count"""
	count = 0
	total_value_counts = np.array([0]*len(count_values or []))
	for _,v in leveldb.iterator():
		count += 1
		
		if count_values is not None:
			array = to_numpy_matrix(v)
			current_count = np.array([0]*len(count_values))
			for i,val in enumerate(count_values):
				current_count[i] = np.sum(array==val)
			total_value_counts += current_count
			
	return total_value_counts, count

def to_numpy_matrix(v):
	""" Convert leveldb value to numpy matrix of shape N x N """
	datum = caffe_pb2.Datum()
	datum.ParseFromString(v)
	# Three cases
	# 1- int imgs in data, 
	# 2- int8 labels in data
	if len(datum.data) > 0:
		type_ = get_data_type(datum)
		matrix = np.fromstring(datum.data, dtype=type_)
	# 3- float imgs in float_data, 
	elif len(datum.float_data) > 0:
		matrix = np.array(datum.float_data)
	else:
		raise ValueError("Serialized datum have empty data and float_data.")
	
	matrix = matrix.reshape((datum.height, datum.width))
	return matrix