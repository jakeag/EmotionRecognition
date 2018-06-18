# PainInputGen
from filetools import find_files
from PIL import Image
import numpy as np
from random import shuffle
import re
# Format

# Categories:
label_types = ['Sad','Happy',  'Pain', 'Neutral', 'Surprised', 'Angry', 'Fear', 'Disgusted']
#               0        1      2       3       4          5     6       7             8
label_form  = ['sa' ,'h'    ,  'p'   , 'n'  , 's'    , 'a',   'f', 'd']
#


def path_to_lab(filename):
	for x in range(len(label_form)):
		# print(filename.split('/')[-1])
		if label_form[x] in filename.split('/')[-1].split('.')[0]:
			# print(label_form[x], x)
			# if x is None:
			# 	print("None",label_form)
			return x

def open_image(filename):
	img = Image.open(filename)
	img = np.asarray(img)
	return img

class InputGenerator:
	def __init__(self):
		directory = 'F:\pain\pain'
		img_ext   = '.jpg'
		self.dir_list = find_files(directory,img_ext)
		lab_list = [path_to_lab(filename) for filename in self.dir_list]
		img_list = [open_image(filename) for filename in self.dir_list]
		self.data_list = [(img_list[x],lab_list[x]) for x in range(len(img_list)) if lab_list[x] is not None]
		# shuffle(self.data_list)

	def get_next_batch(self,batch_size):
		access_list = list(range(len(self.data_list)))
		shuffle(access_list)
		imgs = []
		labs = []
		# [self.data_list[access_list[x]] for x in range(batch_size)]
		[(imgs.append(self.data_list[access_list[x]][0]),labs.append(self.data_list[access_list[x]][1])) for x in range(batch_size)]
		return imgs, labs

def main():
	inpg = InputGenerator()
	for x in range(len(inpg.data_list)):
		image,label = inpg.data_list[x]

		# print(image.shape,label)


main()
