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
import lutils


from tqdm import tqdm, trange
import plyvel
import argparse
import numpy as np
import sys, os
from caffe.proto import caffe_pb2
import png

	


def visualize_img_seg(dbimg, dbseg, outdir, N_start=0, N=40, with_raw_img=False):
	""" Visualize images and their segmentation labels on top.
	The function loops through all images found in the database and keeps visualizing """
	itimg = dbimg.iterator()
	itseg = dbseg.iterator()
	print 'Started Visualize'
	for i in trange(N_start + N):
		try:
			kimg,vimg = itimg.next()
			kseg,vseg = itseg.next()
			
			if i < N_start: #skip
				continue
		except StopIteration:
			break 
		
		img = lutils.to_numpy_matrix(vimg)
		seg = lutils.to_numpy_matrix(vseg)
		#Print histogram of labels
		#print np.where(seg==0)[0].shape, np.where(seg==1)[0].shape, np.where(seg==2)[0].shape
		
		assert img.shape == seg.shape, "Image and Label have different dimensions: %s and %s resp." % (str(img.shape),str(seg.shape))
		# Denormalize image values
		img = lutils.denormalize_img_255(img)
		
		#Convert to Color image (add a channel)
		img = np.expand_dims(img, 2)
		#Fill R,G,B with same value
		img = img.repeat(3,2)
		
		#Build seg image such that :  Liver->Red , Lesion->Green
		# ex: R channel is 255 only for things labeled with Liver
		highlight = np.zeros(img.shape)
		highlight[seg==1, 0] = 80
		highlight[seg==2, 1] = 80
		
		highlight = highlight + img
		np.clip(highlight, 0, 255,highlight)

		writer = png.Writer(img.shape[1], img.shape[0])
		# Write segmentation+image
		file = open(os.path.join(outdir,"Seg_"+kimg+".png"), 'w')
		writer.write(file, np.reshape(highlight, (-1, highlight.shape[0]*3)))
		# Write raw image
		if with_raw_img:
			file = open(os.path.join(outdir,"Img_"+kimg+".png"), 'w')
			writer.write(file, np.reshape(img, (-1, highlight.shape[0]*3)))
	
	
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Reads leveldb and prints some stats")
parser.add_argument("-dbimg", required=True, help="Leveldb path of images")
parser.add_argument("-dbseg", required=True, help="Leveldb path of the corresponding segmentation labels")
parser.add_argument("-o",required=True, help="Output directory to write PNG images to.")
parser.add_argument("-s",default=0, type=int, help="Index of slice to start at.")
parser.add_argument("-n",default=40, type=int, help="Number of slices to save.")
parser.add_argument("--with-raw-img",dest='with_raw_img', default=False, action="store_true", help="Writes also a raw image without overlayed label")
args = parser.parse_args()

# Open database
try:
	dbimg=plyvel.DB(args.dbimg)
	dbseg=plyvel.DB(args.dbseg)
except:
	newimgpath = os.path.join(args.dbimg,"train_img")
	newsegpath = os.path.join(args.dbseg,"train_seg")
	print 'Given path has no leveldb, trying %s , and %s' % (newimgpath, newsegpath)
	dbimg=plyvel.DB(newimgpath)
	dbseg=plyvel.DB(newsegpath)
	
print 'Calling'
visualize_img_seg(dbimg, dbseg, args.o, args.s, args.n, with_raw_img = args.with_raw_img)






