"""A utility program to make train.txt and test.txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

_TEST_SEQ = [5, 17, 19, 20]
_OSM_SEQ = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]


def main():
	"""Main function."""
	osm_file = open("cache/osm.txt", "w+")
	test_file = open("cache/test.txt", "w+")
	dataset = "data/KITTI_Tracking/seg_data"

	for seq in _OSM_SEQ:
		seq_name = str(seq).zfill(4)
		length = len(os.listdir(os.path.join(dataset,seq_name)))
		for file_no in range(18, length, 18):
			osm_file.write("{0}/{1}.png\n".format(seq_name, 
				str(file_no).zfill(6)))

	for seq in _TEST_SEQ:
		seq_name = str(seq).zfill(4)
		length = len(os.listdir(os.path.join(dataset,seq_name)))
		for file_no in range(18, length, 18):
			test_file.write("{0}/{1}.png\n".format(seq_name, 
				str(file_no).zfill(6)))

	osm_file.close()
	test_file.close()


if __name__=='__main__':
	main()
