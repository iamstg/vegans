"""Train WGANGP for Ground Plane occupancy grid prediction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

import transforms as ext_transforms
from models.enet import ENet, ENetDepth
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

# vegan imports
from wgan_gp import WGANGP
from utils import plot_losses

# ENet import
from models.enet import ENet

# Import the datasets
from data import MapLite as dataset
from data import OSMData as real_dataset


# Get the arguments
args = get_arguments()

device = torch.device(args.device)

# Mean color, standard deviation (R, G, B)
color_mean = [0., 0., 0.]
color_std = [1., 1., 1.]

ngpu = 4


def load_dataset(dataset):
	print("\nLoading dataset...\n")

	print("Selected dataset:", args.dataset)
	print("Dataset directory:", args.dataset_dir)
	print('Train file:', args.trainFile)
	print('Val file:', args.valFile)
	print('Test file:', args.testFile)
	print("Save directory:", args.save_dir)

	image_transform = transforms.Compose(
		[transforms.Resize((args.height, args.width)),
		 transforms.ToTensor()])

	label_transform = transforms.Compose([
		transforms.Resize((args.height, args.width)),
		ext_transforms.PILToLongTensor()
	])

	# Get selected dataset
	# Load the training set as tensors
	train_set = dataset(args.dataset_dir, args.trainFile, mode='train', transform=image_transform, \
		label_transform=label_transform, color_mean=color_mean, color_std=color_std)
	train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	# Load the validation set as tensors
	val_set = dataset(args.dataset_dir, args.valFile, mode='val', transform=image_transform, \
		label_transform=label_transform, color_mean=color_mean, color_std=color_std)
	val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	# Load the test set as tensors
	test_set = dataset(args.dataset_dir, args.testFile, mode='inference', transform=image_transform, \
		label_transform=label_transform, color_mean=color_mean, color_std=color_std)
	test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	# Get encoding between pixel valus in label images and RGB colors
	class_encoding = train_set.color_encoding

	# Get number of classes to predict
	num_classes = len(class_encoding)

	# Print information for debugging
	print("Number of classes to predict:", num_classes)
	print("Train dataset size:", len(train_set))
	print("Validation dataset size:", len(val_set))

	# Get a batch of samples to display
	if args.mode.lower() == 'test':
		images, labels = iter(test_loader).next()
	else:
		images, labels = iter(train_loader).next()
	print("Image size:", images.size())
	print("Label size:", labels.size())
	print("Class-color encoding:", class_encoding)

	# Show a batch of samples and labels
	if args.imshow_batch:
		print("Close the figure window to continue...")
		label_to_rgb = transforms.Compose([
			ext_transforms.LongTensorToRGBPIL(class_encoding),
			transforms.ToTensor()
		])
		color_labels = utils.batch_transform(labels, label_to_rgb)
		utils.imshow_batch(images, color_labels)

	# Get class weights from the selected weighing technique
	print("Weighing technique:", args.weighing)
	# If a class weight file is provided, try loading weights from in there
	class_weights = None
	if args.class_weights_file:
		print('Trying to load class weights from file...')
		try:
			class_weights = np.loadtxt(args.class_weights_file)
		except Exception as e:
			raise e
	if class_weights is None:
		print("Computing class weights...")
		print("(this can take a while depending on the dataset size)")
		class_weights = 0
		if args.weighing.lower() == 'enet':
			class_weights = enet_weighing(train_loader, num_classes)
		elif args.weighing.lower() == 'mfb':
			class_weights = median_freq_balancing(train_loader, num_classes)
		else:
			class_weights = None

	if class_weights is not None:
		class_weights = torch.from_numpy(class_weights).float().to(device)
		# Set the weight of the unlabeled class to 0
		print("Ignoring unlabeled class: ", args.ignore_unlabeled)
		if args.ignore_unlabeled:
			ignore_index = list(class_encoding).index('unlabeled')
			class_weights[ignore_index] = 0

	print("Class weights:", class_weights)

	return (train_loader, val_loader,
			test_loader), class_weights, class_encoding


def load_real_data(real_dataset):
	print("\nLoading OSM dataset...\n")

	print("Selected dataset:", args.dataset)
	print("Dataset directory:", args.dataset_dir)
	print('OSM file:', args.osmFile)
	print("Save directory:", args.save_dir)

	image_transform = transforms.Compose(
		[transforms.Resize((args.height, args.width)),
		 transforms.ToTensor()])

	# Get selected dataset
	# Load the osm set as tensors
	osm_set = real_dataset(args.dataset_dir, args.osmFile, mode='train', transform=image_transform, color_mean=color_mean, color_std=color_std)
	osm_loader = data.DataLoader(osm_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	# Print information for debugging
	print("OSM dataset size:", len(osm_set))

	return osm_loader


class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x


def main():
	"""Main function."""

	loaders, class_weights, class_encoding = load_dataset(dataset)
	train_loader, val_loader, test_loader = loaders

	num_classes = len(class_encoding)

	critic = DiscriminativeNet()
	generator = ENet(num_classes)

	dataloader = load_real_data(real_dataset)

	optimizer_D = optim.Adam(critic.parameters(), lr=0.0001, 
		betas=(0.5, 0.999))
	optimizer_G = optim.Adam(generator.parameters(), lr=0.0001,
	 	betas=(0.5, 0.999))

	gan = WGANGP(generator, critic, dataloader, train_loader, test_loader, 
		class_weights, class_encoding, ngpu=ngpu, device=device, nr_epochs=500, print_every=10, save_every = 400, optimizer_D=optimizer_D, optimizer_G=optimizer_G)

	gan.train()
	samples_l, D_losses, G_losses = gan.get_training_results()
	# determine what exactly is samples_l

	
if __name__=='__main__':
	main()