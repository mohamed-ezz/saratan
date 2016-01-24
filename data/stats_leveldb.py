#! /usr/bin/env python
'''
Created on Jan 22, 2016
@author: Mohamed.Ezz

This script introspects leveldb database and print some information about it.
The given leveldb is assumed to be created using the prep_caffe_ds.py script.

'''

import plyvel
import argparse
import numpy as np
import sys, os
from caffe.proto import caffe_pb2

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
	
def find_datatype(leveldb):
	""" Return the numpy type of the pixels stored in the given leveldb. Be it in datum.data or datum.float_data """
	datum = nth_datum(leveldb, 0)
	if len(datum.data) > 0:
		return get_data_type(datum)
	elif len(datum.float_data) > 0:
		return np.float
	else:
		raise ValueError("Serialized datum have empty data and float_data.")	

def find_image_dimension(leveldb):
	_, v = leveldb.iterator().next()
	datum = caffe_pb2.Datum()
	datum.ParseFromString(v)
	
	return datum.height, datum.width, datum.channels

def find_keycount(leveldb):
	""" Takes a plyvel.DB instance and returns number of keys found """
	count = 0
	for _,_ in leveldb.iterator():
		count += 1
	return count

def find_pixel_range(leveldb):
	""" Gets min and max values found in all the matrices (values of the leveldb) """
	_, v = leveldb.iterator().next()
	matrix = to_numpy_matrix(v)
	return matrix.min(), matrix.max() 
	
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Reads leveldb and prints some stats")
parser.add_argument("-db", required=True)
args = parser.parse_args()

# Open database
try:
	db=plyvel.DB(args.db)
except:
	newpath = os.path.join(args.db,"train_img")
	print 'Path have no leveldb, trying %s' % newpath
	db=plyvel.DB(newpath)
	
print "Image dimension : ", find_image_dimension(db)
print "Pixel range     : ", find_pixel_range(db)
print "Data type       :", find_datatype(db)
print "Number of Keys  : ", find_keycount(db)





