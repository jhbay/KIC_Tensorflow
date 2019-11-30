# encoding: UTF-8
# 2018.12 : created by Seungkwon Lee
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


png_file_dir = './cifar10_png/test_small'                   # input image folder
label_file = './cifar10_png/cifar10_label_small.txt'  		# output file 


path_label = []
# read file list in pag_file_dir
for f in os.listdir(png_file_dir) :
	# 파일형 필터링 필수 - 정제작업1.
	if isfile(join(png_file_dir,f)) and f.lower().endswith('.png'):
		fileNameSplit = f.split("_")
		fIndex = fileNameSplit[0]
		fLabel = fileNameSplit[1].split('.')[:-1][0] #.replace(".png","")
		path_label.append((png_file_dir+'/'+f,  label_dic[fLabel]))

# shuffle list
# shuffling 하는 이유는 training부터 일관성 없이 넣어줘야 학습이 편향이 되지않는다.
# 만약 sequence하면서 일정한 데이터세트를 학습시키면 편향(bias)로 학습이 제대로 되지않는경우가 많기떄문.
random.shuffle(path_label)

# write label text file
with open(label_file, 'w') as wfile :
	for item in path_label :
		wfile.write(item[0] + ' ' + item[1] + '\n')
   