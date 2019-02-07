#!/usr/bin/env

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from scipy.io import loadmat

def make_avi(image_stack, output_folder, filename='stack.avi'):
	'''
	makes an avi movie of a three-dimensional numpy array

	'''
	rows = image_stack.shape[1] 
	cols = image_stack.shape[2]
	fourcc = cv2.VideoWriter_fourcc(*'H264')
	# this has to be width (cols), height (rows)
	writer = cv2.VideoWriter(output_folder+filename, fourcc, FRAME_RATE, (cols, rows))

	num_frames = image_stack.shape[0]
	print(num_frames)
	# walk through frame indices
	# this takes advantage of memory-mapping 
	for frame_idx in range(num_frames):
		# convert to 8-bit image (magic?)
		frame = bytescale(image_stack[frame_idx,:,:])
		# Apply the reversed jet colormap like a function to any array:
		cm = plt.get_cmap('jet_r')
		frame = cm(frame)
		# get a three-channel image
		# map 0-->1 to 0-->255
		frame = (frame[:, :, :3]*255).astype('uint8')
		writer.write(frame)
	writer.release()

def get_metadata_from_mat(path):
	############################
	# Read in mat files -- HACKY 
	############################

	#  get timestamps 
	raw_timestamps = loadmat(path+'timeStamps.mat')['vtTimestamps'] 
	stack_timestamps = []
	for trial_timestamps in np.squeeze(raw_timestamps):
		stack_timestamps.append(np.squeeze(trial_timestamps))

	# stack is divided into blocks
	num_blocks = len(stack_timestamps)

	# sizes of each block 
	raw_stack_sizes = np.squeeze(loadmat(path+'allStacksSize.mat')['allStacksSize']) 
	stack_sizes = []
	for image_size in raw_stack_sizes:
		stack_sizes.append(image_size.flatten())
	stack_sizes = np.array(stack_sizes)

	# number of images 
	num_frames = np.sum(stack_sizes[:,0])

	# image size
	frame_cols = stack_sizes[0][2]
	frame_rows = stack_sizes[0][1]

	return (num_frames, frame_rows, frame_cols), num_blocks


if __name__ == '__main__':

	# to-do 
		# load behavior data 
		# compute SVD
		# make video of 500 SVD components and original data 
		# highpass filter SVD data 

	mice = ["t61","t62","t63"]
	mouse_name = mice[0]

	FRAME_RATE = 40 #Hz
	DF_FILE = "dF_stack.bin"
	folder = "/home/spencer/Documents/widepy/data/" + mouse_name + "/"

	stack_shape, num_blocks = get_metadata_from_mat(folder)
	rows = int(stack_shape[1]) # otherwise only 16 bit!
	cols = int(stack_shape[2])
	frames = int(stack_shape[0])

	# r for READ-ONLY -- otherwise use c for copy-on-write
	# open the big stacked dF/F file 
	# memmap the contents -- F is column ordering, C is row ordering 
	file = open(folder+DF_FILE,mode='r')
	stacked_tensor = np.memmap(file, dtype='uint16', mode='r', shape=stack_shape, order='F')

	chunk = stacked_tensor[:100]

	# get each image as a 1D column stack 
	print(chunk.shape)
	chunk = chunk.transpose(0,2,1).reshape((chunk.shape[0],rows*cols,)).transpose(1,0)
	print(chunk.shape)

	# make avi 
	# make_avi(stacked_tensor, "/home/spencer/Documents/widepy/video/", filename=mouse_name+'_stack.avi')

	# one pixel for the whole recording
	plt.plot(stacked_tensor[:100,150,220],'b')
	# plt.show()
	plt.plot(chunk[150+(rows)*(220),:],'r')
	plt.show()
	
	# # make an image of one time point in a random trial 
	# plt.imshow(stacked_tensor[-10,:,:])
	# plt.show()
	# plt.imshow(bytescale(stacked_tensor[0,:,:]))
	 
