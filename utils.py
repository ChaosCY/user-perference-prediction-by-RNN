import numpy as np
import pandas as pd
import codecs
import random
import re
from math import sqrt

def write_file(data_df, output_file):
	data_df.to_csv(output_file, header=None, index=False)

def to_train_test(data_x, data_y, switch_labels, last_label):
	'''得到train和test集
	'''
	train_x = []
	test_x = []
	train_y = []
	test_y = []
	train_s = []
	test_s = []
	train_l = []
	test_l = []
	for x, y, s, l in zip(data_x, data_y, switch_labels, last_label):
		rand = random.randint(0,9)
		if rand < 3:
			test_x.append(x)
			test_y.append(y)
			test_s.append(s)
			test_l.append(l)
		else:
			train_x.append(x)
			train_y.append(y)
			train_s.append(s)
			train_l.append(l)
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	train_s = np.array(train_s)
	train_l = np.array(train_l)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	test_s = np.array(test_s)
	test_l = np.array(test_l)
	return train_x, train_y, train_s, train_l, test_x, test_y, test_s, test_l

def read_data(train_file):
	'''读取数据
	'''
	train_x = []
	users = []
	for line in open(train_file):
		line = re.sub('[\r\n\t]', '', line)
		line = line.split(',')
		if(len(line) >= 4):	
			line = list(map(int, line))
			user = line[0]
			train_x.append(line[1:])
			users.append(user)
	train_x = np.array(train_x)
	users = np.array(users)
	
	return train_x, users
	
def read_feat_data(feature_file):
	'''读取数据
	'''
	features = []
	for line in open(feature_file):
		line = re.sub('[\r\n\t]', '', line)
		line = line.split(',')
		line = list(map(float, line))
		features.append(line[1:])
			
	features = np.array(features)
	
	return features
	
def read_item_labels(item_label_file):
	'''读取产品特征
	'''
	return pd.read_csv(item_label_file)

def get_switch_last_label(data_x):
	''' 获取倒数10个标签改变情况，类似于:[0,0,0,1,0,0,1,0,0,1]
	    以及最后一个标签
	'''
	lengths = np.array([len(s) for s in data_x])
	max_length = max(lengths)
	padding_x_front = np.zeros([len(data_x), max_length])
	for idx,seq in enumerate(data_x):
		padding_x_front[idx, max_length-len(seq):]  = seq
	padding_x_front = padding_x_front[:, max_length-11:]
	switch_labels = padding_x_front[:, 1:] - padding_x_front[:, :10]
	df = pd.DataFrame(switch_labels)
	df[df!=0] = 1
	switch_labels = df.as_matrix()
	last_label = [x[-1] for x in data_x]
	
	return switch_labels, last_label

def data_padding(test_x):
	''' padding
	'''
	lengths = np.array([len(s) for s in test_x])
	max_length = max(lengths)
	padding_x = np.zeros([len(test_x), max_length])
	for idx,seq in enumerate(test_x):
		padding_x[idx, :len(seq)] = seq
	test_x = padding_x
	
	return test_x