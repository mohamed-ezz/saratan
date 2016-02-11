#! /usr/bin/env python
'''
Created on Jan 22, 2016
@author: Mohamed.Ezz

This script reads leveldb database and visualize the slices in different ways
according to cmd arguments

'''
# Add project to search path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plyvel
import argparse
import numpy as np
import sys, os
from caffe.proto import caffe_pb2
import png
import leveldb_utils as ldbutil

	
def denormalize_img(arr):
	""" Denormalizes a nparray to 0-255 values """
	min = arr.min()
	max = arr.max()

	new = (arr - min) * (255.0/(max-min))
	return new.astype(np.uint8)

def visualize_img_seg(dbimg, dbseg, outdir, N_start=0, N=40):
	""" Visualize images and their segmentation labels on top.
	The function loops through all images found in the database and keeps visualizing """
	itimg = dbimg.iterator()
	itseg = dbseg.iterator()
	print 'Started Visualize'
	for i in range(N_start + N):
		try:
			kimg,vimg = itimg.next()
			kseg,vseg = itseg.next()
			
			if i < N_start: #skip
				continue
		except StopIteration:
			break 
		
		print 'processing image', i
		img = ldbutil.to_numpy_matrix(vimg)
		seg = ldbutil.to_numpy_matrix(vseg)
		#Print histogram of labels
		print np.where(seg==0)[0].shape, np.where(seg==1)[0].shape, np.where(seg==2)[0].shape
		
		assert img.shape == seg.shape, "Image and Label have different dimensions: %s and %s respect." % (str(img.shape),str(seg.shape))
		# Denormalize image values
		img = denormalize_img(img)
		
		#Convert to Color image (add a channel)
		img = np.expand_dims(img, 2)
		#Fill R,G,B with same value
		img = img.repeat(3,2)
		
		#Build seg image such that :  Liver->Red , Lesion->Green
		# ex: R channel is 255 only for things labeled with Liver
		highlight = np.zeros(img.shape)
		highlight[seg==1, 0] = 255
#		print highlight
#		IPython.embed()
		highlight[seg==2, 1] = 255
		
		print img.min(), img.max()
		highlight = highlight + img
		np.clip(highlight, 0, 255,highlight)
		writer = png.Writer(img.shape[1], img.shape[0])
		file = open(os.path.join(outdir,"Seg_"+kimg+".png"), 'w')
		writer.write(file, np.reshape(highlight, (-1, highlight.shape[0]*3)))
		file = open(os.path.join(outdir,"Img_"+kimg+".png"), 'w')
		#IPython.embed()
		writer.write(file, np.reshape(img, (-1, highlight.shape[0]*3)))
	
	
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Reads leveldb and prints some stats")
parser.add_argument("-dbimg", required=True, help="Leveldb path of images")
parser.add_argument("-dbseg", required=True, help="Leveldb path of the corresponding segmentation labels")
parser.add_argument("-o",required=True, help="Output directory to write PNG images to.")
parser.add_argument("-s",default=0, type=int, help="Index of slice to start at.")
parser.add_argument("-n",default=40, type=int, help="Number of slices to save.")
args = parser.parse_args()

# Open database
try:
	dbimg=plyvel.DB(args.dbimg)
	dbseg=plyvel.DB(args.dbseg)
except:
	newimgpath = os.path.join(args.db,"train_img")
	newsegpath = os.path.join(args.db,"train_seg")
	print 'Path have no leveldb, trying %s , and %s' % (newimgpath, newsegpath)
	dbimg=plyvel.DB(newimgpath)
	dbseg=plyvel.DB(newsegpath)
	
print 'Calling'
visualize_img_seg(dbimg, dbseg, args.o, args.s, args.n)






