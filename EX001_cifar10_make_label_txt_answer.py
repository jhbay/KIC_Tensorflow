# encoding: UTF-8
# 2018.12 : created by Seungkwon Lee(kahnlee@naver.com)

import os
from os.path import isfile, join
import random

label_dic = {'airplane'		: '0',
			 'automobile'	: '1',
			 'bird'			: '2',
			 'cat'			: '3',
			 'deer'			: '4',
			 'dog'			: '5',
			 'frog'			: '6',
			 'horse'		: '7',
			 'ship'			: '8',
			 'truck'		: '9'  }

# image folder & label file
# png_file_dir = './cifar10_png/train'
# label_file = './cifar10_png/cifar10_label_train.txt'

# png_file_dir = './cifar10_png/test'
# label_file = './cifar10_png/cifar10_label_test.txt'

png_file_dir = './cifar10_png/test_small'
label_file = './cifar10_png/cifar10_label_small.txt'

path_label = []
for f in os.listdir(png_file_dir) :
	if isfile(join(png_file_dir, f)) and f.lower().endswith('.png')	:  	# 90_airplane.png
		fn = f.split('.')[0]      										# 90_airplane
		path_file = png_file_dir + '/' + f   							# join(png_file_dir,f) 사용시 '\' 문제 있음 
		label = fn.split('_')[1]										# airplane

		path_label.append((path_file, label_dic[label]))

# shuffle list
random.shuffle(path_label)

# write 'cifar10_train.txt'
with open(label_file, 'w') as wfile :
	for item in path_label :
		wfile.write(item[0] + ' ' + item[1] + '\n')