from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import OrderedDict
import numpy as np
import imageio
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from . import utils


class OSMData(data.Dataset):
	"""MapLite dataset

	Keyword arguments:
	- root_dir (``string``): Path to the base directory of the dataset
	- list_file (``string``): Path to file containing a list of images to be loaded
	- transform (``callable``, optional): A function/transform that takes in a 
	PIL image and returns a transformed version of the image. Default: None.
	- label_transform (``callable``, optional): A function/transform that takes 
	in the target and transforms it. Default: None.
	- loader (``callable``, optional): A function to load an image given its path.
	By default, ``default_loader`` is used.
	- color_mean (``list``): A list of length 3, containing the R, G, B channelwise mean.
	- color_std (``list``): A list of length 3, containing the R, G, B channelwise standard deviation.
	- seg_classes (``string``): The palette of classes that the network should learn.
	"""

	def __init__(self, root_dir, list_file, mode='train', transform=None, label_transform = None, \
		loader=utils.maplite_loader, color_mean=[0.,0.,0.], color_std=[1.,1.,1.]):
		
		self.root_dir = root_dir
		self.list_file = list_file
		self.mode = mode
		self.transform = transform
		self.label_transform = label_transform
		self.loader = loader
		self.length = 0
		self.color_mean = color_mean
		self.color_std = color_std

		# Get the list of scenes, and generate paths
		image_list = []
		try:
			list_file = open(self.list_file, 'r')
			scenes = list_file.readlines()
			list_file.close()
			for scene in scenes:
				scene = scene.strip().split()
				image_list.append(scene[0])
		except Exception as e:
			raise e

		# Get osm data filepaths
		self.data = []
		for img_id in image_list:
			self.data.append(os.path.join(self.root_dir, 'osm_label', str(img_id)))
			self.length += 1


	def __getitem__(self, index):
		""" Returns element at index in the dataset.

		Args:
		- index (``int``): index of the item in the dataset

		Returns:
		A tuple of ``PIL.Image`` (image, label) where label is the ground-truth of the image

		"""

		data_path = self.data[index]
		# Load image
		img = np.array(imageio.imread(data_path)).astype(np.uint8)
		label =  0
		if self.mode.lower() == 'inference':
			return img, label, self.data[index], self.labels[index]
		else:
			return img, label


	def __len__(self):
		""" Returns the length of the dataset. """
		return self.length
