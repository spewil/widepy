import os
import shutil
import re 
import glob
import bz2


# User input--directories
# baseDir='/mnt/microscopy/projects/KeCa_20141001_BMI_widefield/WF_imaging/reduced/t41/20160710/rs/stim/'; % visual stimulus happening 
# baseDir='/mnt/microscopy/projects/KeCa_20141001_BMI_widefield/WF_imaging/reduced/t35/30260603/retrosplenial/spon-pre/'; % spontaneous in the dark pre-task
# baseDir = '/mnt/microscopy/projects/KeCa_20141001_BMI_widefield/WF_imaging/reduced/t64/rs/spon/' # spontaneous in the dark
# saveDir = baseDir + 'analyzed/'
# source_img = baseDir + 'imaging/'
# dFfile = source_img + 'dF/DF-new3.bin'

# # change to working directory 
# os.chdir(source_img)

mice = ["t60","t61","t62","t63","t64","t65"]

###########
### RAW ###
###########

# raw_dest = '/home/spencer/Documents/widefield_analysis/data/raw'
# if not os.path.isdir(raw_dest):
# 	os.mkdir(raw_dest)

# raw_pre = '/mnt/microscopy/projects/KeCa_20141001_BMI_widefield/WF_imaging/raw/'
# raw_post = '/rs/stim1/'
# r_name = '-rs-mm@****.tif.bz2' 

# pre + mouse + / + post + mouse + regex_filename

# for mouse in [mice[-4]]:
# 	full_dir = raw_pre+mouse+raw_post
# 	# print(full_dir)
# 	files =	glob.glob(full_dir+mouse+'-rs' + '*.tif.bz2')
# 	# all the raw filenames 
# 	for file in files: 
# 		print(file)
# 	    # shutil.copy(full_file_name, dest)

# os.rename('a.txt', 'b.kml')

###############
### REDUCED ###
###############

reduced_dest = '/home/spencer/Documents/widefield_analysis/data/'

# reduced data:
# reduced_pre = '/mnt/microscopy/projects/KeCa_20141001_BMI_widefield/WF_imaging/reduced/' 
# reduced_post = '/rs/stim1/imaging/'
# reduced_name = '-rs-stim1-1@****_binn_perm.bin.bz2'

for mouse in mice[4:]:
	folder = reduced_dest + mouse + '/'
	os.chdir(folder)
	# all the raw zipped filenames 
	files =	glob.glob('*.bin.bz2')
	newfile = folder + "dF_stack.bin"
	for file in files: 
		with bz2.BZ2File(folder+file) as zipped, open(newfile, 'wb') as unzipped:
			print('unzipping: ',folder+file)
			print('writing to:',newfile)
			shutil.copyfileobj(zipped,unzipped,length=16*1024*1024)
		print('decompressed')
		print('removing:',file)
		os.remove(file)

# pre + mouse + / + post + mouse + regex_filename

# full_stack_name = 'dF/DF-new3.bin.bz2'
# for mouse in mice:
# 	reduced_dest = reduced_dest + mouse + '/'
# 	if not os.path.isdir(reduced_dest):
# 		os.mkdir(reduced_dest)
# 	full_dir = reduced_pre+mouse+reduced_post
# 	print(full_dir)
# 	files =	glob.glob(full_dir+mouse+'-rs*'+'@'+'*.bin.bz2')
# 	# all the raw filenames 
# 	for file in files: 
# 		print(file)
# 		shutil.copy(file, reduced_dest)
# pre + full_stack_name

metadata_names = ['timeStamps.mat',
'allStacks.mat',
'allStacksSize.mat',
'globalDf2.mat',
'globalFluo.mat',
'maskX_new.mat',
'maskY_new.mat',
'recordMask.mat',
'roiFluo.mat',
'ROI_mask.mat']

# # behavioral: 
# behavior_pre = '/mnt/microscopy/projects/KeCa_20141001_BMI_widefield/WF_imaging/behavior'
# filename = 'manExpID_###_behav.bin' 
# behavior_dest = '/home/spencer/Documents/widefield_analysis/data/behavior'
# if not os.path.isdir(behavior_dest):
# 	os.mkdir(behavior_dest)
# meta_dest = '/home/spencer/Documents/widefield_analysis/data/meta'
# if not os.path.isdir(meta_dest):
# 	os.mkdir(meta_dest)


