#!/usr/bin/env

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.misc import bytescale

# mice = ["t60","t61","t62","t63","t64","t65"]
mice = ["t61"]

frame_rate = 40 #Hz

metadata_names = [
'timeStamps.mat',
'allStacks.mat',
'allStacksSize.mat',
'globalDf2.mat',
'globalFluo.mat',
'maskX_new.mat',
'maskY_new.mat',
'recordMask.mat',
'roiFluo.mat',
'ROI_mask.mat'
]

os.chdir("/home/spencer/Documents/widefield_analysis/data/")
os.chdir(mice[0])

# Read in mat files -- HACKY 
############################

# timestamps are 115 lists of 1063 floats
raw_timestamps = loadmat('timeStamps.mat')['vtTimestamps'] 
stack_timestamps = []
for trial_timestamps in np.squeeze(raw_timestamps):
	# print(type(np.squeeze(trial_time_stamps)))
	stack_timestamps.append(np.squeeze(trial_timestamps))

print('stack_timestamps: ', len(stack_timestamps))

maskPixelsX = np.array(loadmat('maskX_new.mat')['maskPixelsX']).flatten() # mask pixel x indices
maskPixelsY = np.array(loadmat('maskY_new.mat')['maskPixelsY']).flatten() # mask pixel y indices

# sizes of each trial tensor, 115 lists of 3 ints 
raw_stack_sizes = np.squeeze(loadmat('allStacksSize.mat')['allStacksSize']) # size of each trial
stack_sizes = []
for image_size in raw_stack_sizes:
	stack_sizes.append(tuple(image_size.flatten()))
print(raw_stack_sizes[0])

#############################


# get names of binary files
# bin_files = [] 
# current_folder = os.getcwd() # current working directory
# for file in os.listdir(current_folder):
#     if file.endswith(".bz2"):
#         bin_files.append(file)
# bin_files.sort()
# bin_files = np.roll(bin_files,1) # move last to first position


# r for READ-ONLY -- otherwise use c for copy-on-write
# open the big stacked dF/F file 
file = open(dFfile,mode='r')
# memmap the contents
stacked_tensor = np.memmap(file, dtype='uint16', mode='r',shape=stacked_tensor_size,order='F')

# WOULD NEED TO UNZIP BZ2 FILES BEFORE MAPPING!!!
# use a memory-mapped tensor version for each data file
# each trial here is it's own memmap that we can treat as an ndarray
# tensor_list = [] 
# for file, size in zip(bin_files, stack_sizes):
# 	print(size,file)
# 	tensor_list.append(np.memmap(file, dtype='uint16', mode='r', shape=size, order='F'))

# one pixel for the whole recording
# plt.plot(stacked_tensor[:,150,200])

# make an image of one time point in a random trial 
plt.imshow(stacked_tensor[0,:,:])
# plt.show()
# plt.imshow(bytescale(stacked_tensor[0,:,:]))

# Get the color map by name:
cm = plt.get_cmap('jet_r')

# OPENCV SOLUTION -- unclear if working
# make a movie of the first trial 
os.chdir('/home/spencer/Documents/widefield_analysis/')
folder = os.path.join(os.getcwd(),'video')
if not os.path.exists(folder):
	os.mkdir(folder)
fourcc = cv2.VideoWriter_fourcc(*'H264')
# this has to be width (cols), height (rows)
writer = cv2.VideoWriter(folder+'/test2.avi', fourcc, 40, (373,300))
trial_size = stack_sizes[0]
for frame_idx in range(stack_sizes[0][0]):
	# convert to 8-bit image (magic?)
	frame = bytescale(stacked_tensor[frame_idx,:,:])
	# Apply the colormap like a function to any array:
	frame = cm(frame)
	# get a three-channel image
	# map 0-->1 to 0-->255
	frame = (frame[:, :, :3]*255).astype('uint8')
	writer.write(frame)
writer.release()

plt.show()

# to do 
	# look at smoothing, correction for bleaching/drift
	# look at Ca++ python pipelines 