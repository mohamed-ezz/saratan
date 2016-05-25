'''
Contains common functions for reading data out of leveldb

@author: Mohamed.Ezz
'''
import plyvel, lmdb
import numpy as np

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

def denormalize_img_255(arr):
	""" Denormalizes a nparray to 0-255 values """
	min = arr.min()
	max = arr.max()

	new = (arr - min) * (255.0/(max-min))
	return new.astype(np.uint8)

def leveldb_arrays(leveldbdir):
	""" Generator. Given leveldb directory, iterate the stored data as numpy arrays. Yields (Key, NumpyArray) """
	db = CaffeDatabase(leveldbdir)
	for k,v in db.iterator():
		yield k, to_numpy_matrix(v)

def nth_datum(caffedb, n):
	""" Returns nth datum. 0-based index"""
	n+=1
	it = caffedb.iterator()
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

def find_keycount(caffedb, count_values=None):
	""" Takes a CaffeDatabase or plyvel.DB instance and returns number of keys found and count of each value. 
	count_values is a list of values to count, e.g. count_values=[0,1,2] will return [count of 1s, count of 2s, count of 3s]
	if count_values is None, return value of this function is [],key_count"""
	count = 0
	total_value_counts = np.array([0]*len(count_values or []))
	for _,v in caffedb.iterator():
		count += 1
		
		if count_values is not None:
			array = to_numpy_matrix(v)
			current_count = np.array([0]*len(count_values))
			for i,val in enumerate(count_values):
				current_count[i] = np.sum(array==val)
			total_value_counts += current_count
			
	return total_value_counts, count

def to_numpy_matrix(v):
	""" Convert leveldb/lmdb value to numpy matrix of shape N x N """
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

def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
	""" Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
	# calc min and max
	min,max = np.amin(arr), np.amax(arr)
	arr = arr.astype(IMG_DTYPE)
	if min <= 0:
		arr = np.clip(arr, min * c_min, max * c_max)
		# right shift to zero
		arr = np.abs(min * c_min) + arr
	else:
		arr = np.clip(arr, min, max * c_max)
		# left shift to zero
		arr = arr - min
	# normalization
	norm_fac = np.amax(arr)
	if norm_fac != 0:
		norm = np.divide(
				np.multiply(arr,255),
			 	np.amax(arr))
	else:  # don't divide through 0
		norm = np.multiply(arr, 255)

	norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
	return norm

def norm_hounsfield_stat(arr, c_min=-100, c_max=200):
    min = np.amin(arr)

    arr = np.array(arr, dtype=IMG_DTYPE)

    if min <= 0:
        # clip
        c_arr = np.clip(arr, c_min, c_max)

        # right shift to zero
        slc_0 = np.add(np.abs(min), c_arr)
    else:
        # clip
        c_arr = np.clip(arr, c_min, c_max)

        # left shift to zero
        slc_0 = np.subtract(c_arr, min)

    # normalization
    norm_fac = np.amax(slc_0)
    if norm_fac != 0:
        norm = np.divide(
            np.multiply(
                slc_0,
                255
            ),
            np.amax(slc_0)
        )
    else:  # don't divide through 0
        norm = np.multiply(slc_0, 255)

	norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
    return norm

class CaffeDatabase():
	""" Abstraction layer over lmdb and leveldb """
	def __init__(self, path, backend='lmdb'):
		self.backend = backend
		assert backend in ['lmdb','leveldb'], "Database backend not known :%s"%backend
		
		if backend=='lmdb':
			self.db = lmdb.open(path)
		elif backend=='leveldb':
			self.db = plyvel.DB(path)
			
	def iterator(self):
		if self.backend == 'lmdb':
			txn = self.db.begin()
			cursor = txn.cursor()
			it = cursor.iternext()
		elif self.backend == 'leveldb':
			it = self.db.iterator()
		return it
		

		
