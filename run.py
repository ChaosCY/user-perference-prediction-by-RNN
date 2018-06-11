import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import codecs
import random

from model import Model
from batch_loader import BatchLoader
import utils

tf.app.flags.DEFINE_string('mode', None, 'validaion or train or predict')
FLAGS = tf.app.flags.FLAGS

def main():
	# setting parameters
	parser = argparse.ArgumentParser(
						formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--num_factors', type=float, default=5,
						help='embedding size')
	parser.add_argument('--model', type=str, default='mlp_bpr',
						help='model')
	parser.add_argument('--epoches', type=str, default=300,
						help='epoches')
	parser.add_argument('--learning_rate', type=float, default=0.01,
						help='learning rate')
	parser.add_argument('--reg_lambda', type=float, default=0.1,
						help='l2_regularizer lambda')
	parser.add_argument('--layers', nargs='?', default='[10,1]',
						help="Size of each layer.")
	parser.add_argument('--batch_size', type=int, default=10000,
						help='minibatch size')
	parser.add_argument('--recom_mode', type=str, default='p_u',
						help='recommendation mode, u_p: users to items, p_u: items to users')
	parser.add_argument('--decay_rate', type=float, default=0.99,
						help='decay rate for Adam')
	parser.add_argument('--keep_prob', type=float, default=0.8,
						help='dropout probility')
	parser.add_argument('--uti_k', type=int, default=30,
						help='top-k recommendation for recommending items to user')
	parser.add_argument('--itu_k', type=int, default=100,
						help='top-k recommendation for recommending users to item')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='directory to store tensorboard logs')
	parser.add_argument('--mode', type=str, default='validation',
						help='train: only train the model, validation: train the model and test it with test data, predict: predict new data')
	parser.add_argument('--dev', type=str, default='cpu',
						help='training by CPU or GPU, input cpu or gpu:0 or gpu:1 or gpu:2 or gpu:3')
	parser.add_argument('--pIter', type=int, default=2,
						help='how many rounds of iterations show the effect on the test set')
	args = parser.parse_args()
	
	if FLAGS.mode=='train':
		# read data from file
		args.num_durations = 14
		args.num_risks = 4
		args.num_returns = 12
		args.num_amounts = 8
		args.num_locations = 3
		args.num_users = 19284
		
		train_file = 'data/mid/itemUnion_distinct'
		data, users = utils.read_data(train_file)
		
		user_file = 'data/mid/itemUnion_distinct_days'
		user_feat = utils.read_feat_data(user_file)
		
		item_label_file = 'data/mid/item_features_no_one-hot-no-normalization.csv'
		labels_df = utils.read_item_labels(item_label_file)
		
		data_x = [x[:-1] for x in data]
		data_y = [x[-1] for x in data]
		
		switch_labels, last_label = utils.get_switch_last_label(data_x)
		
		# split train set and test set
		train_x, train_y, train_s, train_l, test_x, test_y, test_s, test_l = utils.to_train_test(data_x, data_y, switch_labels, last_label)
		lengths = np.array([len(s) for s in test_x])
		# padding data
		test_x = utils.data_padding(test_x)
		
		# generate batches
		batch_loader = BatchLoader(args.batch_size, train_x, train_y, train_s, train_l)
		
		model = Model(args, batch_loader)
		
		# train and save models
		model.train(labels_df, test_x, test_y, test_s, test_l, lengths)
		# save the user and index
		# utils.write_file(user_index_map['user'], "save/user_index_map")
	else:
		print('incorrect mode input...')
	

if __name__ == '__main__':
	main()