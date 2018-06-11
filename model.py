import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import codecs
import random
import os
from tensorflow.python import debug as tf_debug

import utils
import basic_lstm
import evaluation


class Model(object):
	''' train the model and make prediction through the model
	'''
	def __init__(self, args, batch_loader=None):
		self.args = args
		self.batch_loader = batch_loader
		self.model = basic_lstm.BASIC_LSTM(args)
	
	def train(self, labels_df, test_x, test_y, test_s, test_l, lengths):
		''' train
		''' 
		saver = tf.train.Saver()
		with tf.Session() as sess:
			with tf.device("/" + str(self.args.dev)):
				summaries = tf.summary.merge_all()
				writer = tf.summary.FileWriter(os.path.join(self.args.log_dir))
				writer.add_graph(sess.graph)
				
				sess.run(tf.global_variables_initializer())
				
				# product category convert to label
				test_duration_x, test_risk_x, test_return_x, test_amount_x, test_location_x = self.match_labels(test_x, labels_df)
				test_duration_y, test_risk_y, test_return_y, test_amount_y, test_location_y = self.match_labels(test_y, labels_df)
				test_duration_l, test_risk_l, test_return_l, test_amount_l, test_location_l = self.match_labels(test_l, labels_df)
				test_y = np.stack((test_duration_y, test_risk_y, test_return_y, test_amount_y, test_location_y), axis=1)
				test_l = np.stack((test_duration_l, test_risk_l, test_return_l, test_amount_l, test_location_l), axis=1)
				
				total_labels = len(test_y)
				correct_labels = 0
				for i in range(len(test_y)):
					if (test_l[i]==test_y[i]).all():
						correct_labels += 1
				accuracy = 100.0 * correct_labels / float(total_labels)
				# the accuracy recommedding last label
				print("test accuracy: %2.6f" % (accuracy))
				
				list1 = []
				list2 = list(test_duration_y)
				# iterate epoches
				for epoch in range(self.args.epoches):
					# learning rate decay
					sess.run(tf.assign(self.model.lr, self.args.learning_rate * (self.args.decay_rate ** epoch)))
					# reset the batch pointer to 0
					self.batch_loader.reset_batch_pointer()
					# iterate every batches
					for iteration in range(self.batch_loader.num_batches):
						# get the training data
						x, y, s, l, seq_len = self.batch_loader.next_batch()
						
						# trans labels
						duration_x, risk_x, return_x, amount_x, location_x = self.match_labels(x, labels_df)
						duration_y, risk_y, return_y, amount_y, location_y = self.match_labels(y, labels_df)
						duration_l, risk_l, return_l, amount_l, location_l = self.match_labels(l, labels_df)
						# feed data into model
						feed = {self.model.input_duration: duration_x, self.model.targets_duration: duration_y, self.model.last_duration: duration_l, 
								self.model.input_risk: risk_x, self.model.targets_risk: risk_y, self.model.last_risk: risk_l, 
								self.model.input_return: return_x, self.model.targets_return: return_y, self.model.last_return: return_l, 
								self.model.input_amount: amount_x, self.model.targets_amount: amount_y, self.model.last_amount: amount_l, 
								self.model.input_location: location_x, self.model.targets_location: location_y, self.model.last_location: location_l, 
								self.model.switch: s, self.model.seq_len: seq_len, self.model.keep_prob: self.args.keep_prob}
						
						seq, _, loss_epoch = sess.run([self.model.predict_sequence, self.model.optimizer, self.model.loss], feed)
						#print(seq)
						print("epoches: %3d, train loss: %2.6f" % (epoch, loss_epoch))
					
					# feed test data into model
					feed = {self.model.input_duration: test_duration_x, self.model.last_duration: test_duration_l, self.model.input_risk: test_risk_x, self.model.last_risk: test_risk_l, 
							self.model.input_return: test_return_x, self.model.last_return: test_return_l, self.model.input_amount: test_amount_x, self.model.last_amount: test_amount_l, 
							self.model.input_location: test_location_x, self.model.last_location: test_location_l, self.model.switch: test_s, self.model.seq_len: lengths, self.model.keep_prob: 1.0}
					pred = sess.run([self.model.predict_sequence], feed)
					list1 = list(pred)
					
					total_labels = len(test_y)
					correct_labels = 0
					for i in range(len(test_y)):
						if (pred[i]==test_y[i]).all():
							correct_labels += 1
					accuracy = 100.0 * correct_labels / float(total_labels)
					# calculating the accuracy
					print("test accuracy: %2.6f" % (accuracy))
				# save model
				save_path = saver.save(sess, "save/model.ckpt")
				with codecs.open('data/predict/predictions', 'w', encoding = 'utf-8') as f:
					f.write(('\t').join(map(str, list1))+"\n")
					f.write(('\t').join(map(str, list2))+"\n")
					f.write("\n")
	
	def match_labels(self, data, labels_df):
		'''convert the product category to labels
		'''
		shape = data.shape
		data = np.reshape(data, (-1))
		df = pd.DataFrame(data, columns=['cate_id']).astype('int')
		merge_df = pd.merge(df, labels_df, left_on='cate_id', right_on='cate_id', how='left')
		
		df_duration = merge_df['duration1'].fillna(0).as_matrix()
		df_risk = merge_df['risk1'].fillna(0).as_matrix()
		df_return = merge_df['return1'].fillna(0).as_matrix()
		df_amount = merge_df['amount1'].fillna(0).as_matrix()
		df_location = merge_df['location1'].fillna(0).as_matrix()
		
		data_duration = np.reshape(df_duration, shape)
		data_risk = np.reshape(df_risk, shape)
		data_return = np.reshape(df_return, shape)
		data_amount = np.reshape(df_amount, shape)
		data_location = np.reshape(df_location, shape)
		return data_duration, data_risk, data_re