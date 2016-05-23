#! /usr/bin/env python
'''
Created on Jan 22, 2016
@author: Mohamed.Ezz

This script introspects leveldb database and print some information about it.
The given leveldb is assumed to be created using the prep_caffe_ds.py script.

'''
import sys, os
from saratan_utils import CaffeDatabase
projdir =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(projdir)

import plyvel, lmdb
import argparse
import numpy as np
from caffe.proto import caffe_pb2
import saratan_utils
	
def find_datatype(leveldb):
	""" Return the numpy type of the pixels stored in the given leveldb. Be it in datum.data or datum.float_data """
	datum = saratan_utils.nth_datum(leveldb, 0)
	if len(datum.data) > 0:
		return saratan_utils.get_data_type(datum)
	elif len(datum.float_data) > 0:
		return np.float
	else:
		raise ValueError("Serialized datum have empty data and float_data.")	

def find_image_dimension(leveldb):
	_, v = leveldb.iterator().next()
	datum = caffe_pb2.Datum()
	datum.ParseFromString(v)
	
	return datum.height, datum.width, datum.channels



def find_pixel_range(leveldb, n_slices=100):
	""" Gets min and max values found in all the matrices (values of the leveldb) """
	
	it = leveldb.iterator()
	minv =  999999
	maxv = -999999
	for i in range(n_slices):
		_, v = it.next()
		matrix = saratan_utils.to_numpy_matrix(v)
		minv = min(minv, matrix.min())
		maxv = max(maxv, matrix.max())
	return minv, maxv
	
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Reads leveldb and prints some stats")
parser.add_argument("-db", required=True)
parser.add_argument("-n", default = 10, type=int, help="Number of slices to involve in calculations. Default=10")
parser.add_argument("-hist", default=None, nargs='+', type=int, help="list of values to count number of occurences for. e.g. -hist 0 1 2")
parser.add_argument("-backend",default='lmdb')
args = parser.parse_args()

# Open database
try:
	db=CaffeDatabase(args.db, args.backend)
except:
	newpath = os.path.join(args.db,"train_img")
	print 'Path have no leveldb, trying %s' % newpath
	db=CaffeDatabase(args.db, args.backend)
	
print "Image dimension : ", find_image_dimension(db)
print "Pixel range     : ", find_pixel_range(db)
print "Data type       :", find_datatype(db)
hist, keycount = saratan_utils.find_keycount(db,args.hist)
print "Number of Keys  : ", keycount
print "Value histogram :", str(hist)




