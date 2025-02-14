from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser


def get_arguments():
	"""Defines command-line arguments, and parses them.

	"""
	parser = ArgumentParser()

	# Execution mode
	parser.add_argument(
		"--mode",
		"-m",
		choices=['train', 'test', 'inference', 'full'],
		default='train',
		help=("train: performs training and validation; test: tests the model; "
			  "inference: similar to test, but has advanced functionality, eg. saving results, etc."
			  "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
			  "full: combines train and test modes. Default: train"))
	parser.add_argument(
		"--resume",
		action='store_true',
		help=("The model found in \"--checkpoint_dir/--name/\" and filename "
			  "\"--name.h5\" is loaded."))
	parser.add_argument(
		'--generate-images',
		action='store_true',
		help="Used in inference mode. Generates segmentation images, for visualization."
		)

	# Segmentation class variants to use
	parser.add_argument(
		'--seg-classes',
		choices=['nyu40', 'scannet20'],
		default='nyu40',
		help='Choose the palette of classes learnt by the network.')

	# Hyperparameters
	parser.add_argument(
		"--batch-size",
		"-b",
		type=int,
		default=8,
		help="The batch size. Default: 8")
	parser.add_argument(
		"--epochs",
		type=int,
		default=50,
		help="Number of training epochs. Default: 50")
	parser.add_argument(
		"--learning-rate",
		"-lr",
		type=float,
		default=5e-4,
		help="The learning rate. Default: 5e-4")
	parser.add_argument(
		'--beta0',
		type=float,
		default=0.9,
		help='betas[0] for Adam Optimizer. Default: 0.9')
	parser.add_argument(
		'--beta1',
		type=float,
		default=0.999,
		help='betas[1] for Adam Optimizer. Default: 0.999')
	parser.add_argument(
		"--lr-decay",
		type=float,
		default=0.1,
		help="The learning rate decay factor. Default: 0.5")
	parser.add_argument(
		"--lr-decay-epochs",
		type=int,
		default=100,
		help="The number of epochs before adjusting the learning rate. "
		"Default: 100")
	parser.add_argument(
		"--weight-decay",
		"-wd",
		type=float,
		default=2e-4,
		help="L2 regularization factor. Default: 2e-4")

	# Dataset
	parser.add_argument(
		"--dataset",
		choices=['KITTI Tracking', 'SYNTHIA'],
		default='KITTI Tracking',
		help="Dataset to use. Default: KITTI Tracking")
	parser.add_argument(
		"--dataset-dir",
		type=str,
		default="data/KITTI_Tracking",
		help="Path to the root directory of the selected dataset. "
		"Default: data/KITTI_Tracking")
	parser.add_argument(
		"--trainFile",
		type=str,
		default="cache/train.txt",
		help="Path to txt file containing a list of training scenes."
		"Default: cache/train.txt")
	parser.add_argument(
		"--valFile",
		type=str,
		default="cache/test.txt",
		help="Path to txt file containing a list of validation scenes."
		"Default: cache/test.txt")
	parser.add_argument(
		"--testFile",
		type=str,
		default="cache/test.txt",
		help="Path to txt file containing a list of testing scenes."
		"Default: cache/test.txt")
	parser.add_argument(
		"--osmFile",
		type=str,
		default="cache/osm.txt",
		help="Path to txt file containing a list of OSM scenes."
		"Default: cache/osm.txt")
	parser.add_argument(
		"--height",
		type=int,
		default=100,
		help="The image height. Default: 100")
	parser.add_argument(
		"--width",
		type=int,
		default=100,
		help="The image width. Default: 100")
	parser.add_argument(
		"--weighing",
		choices=['enet', 'mfb', 'none'],
		default='ENet',
		help="The class weighing technique to apply to the dataset. "
		"Default: enet")
	parser.add_argument(
		'--class-weights-file',
		type=str,
		help='Path to class weights file. Will skip class weight computation if provided.')
	parser.add_argument(
		"--with-unlabeled",
		dest='ignore_unlabeled',
		action='store_false',
		help="The unlabeled class is not ignored.")

	# Settings
	parser.add_argument(
		"--workers",
		type=int,
		default=4,
		help="Number of subprocesses to use for data loading. Default: 4")
	parser.add_argument(
		"--print-step",
		type=int,
		help="Print loss every step")
	parser.add_argument(
		"--imshow-batch",
		action='store_true',
		help=("Displays batch images when loading the dataset and making "
			  "predictions."))
	parser.add_argument(
		"--device",
		default='cuda',
		help="Device on which the network will be trained. Default: cuda")

	# Storage settings
	parser.add_argument(
		"--name",
		type=str,
		default='ENet',
		help="Name given to the model when saving. Default: ENet")
	parser.add_argument(
		"--save-dir",
		type=str,
		default='save',
		help="The directory where models are saved. Default: save")
	parser.add_argument(
		'--validate-every',
		type=int,
		default=1,
		help='Number of epochs after which to validate')

	return parser.parse_args()
