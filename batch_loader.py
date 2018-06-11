import numpy as np
import math

class BatchLoader(object):
	'''
	generate batches, return every batch
	'''
	def __init__(self, batch_size, train_x, train_y, train_s, train_l):
		self.batch_size = batch_size
		self.create_batches(train_x, train_y, train_s, train_l)
		
	def create_batches(self, train_x, train_y, train_s, train_l):
		'''
		generate batches
		'''
		self.num_batches = math.ceil(len(train_x)/self.batch_size)
		
		if self.num_batches == 0:
			assert False, "Not enough data."
		
		num_samples = len(train_x)
		indexs = np.arange(num_samples)
		np.random.shuffle(indexs)
		
		x = train_x[indexs]
		y = train_y[indexs]
		s = train_s[indexs]
		l = train_l[indexs]
		self.x_batches = np.array_split(x, self.num_batches)
		self.y_batches = np.array_split(y, self.num_batches)
		self.s_batches = np.array_split(s, self.num_batches)
		self.l_batches = np.array_split(l, self.num_batches)
		self.batch_maxlen = []
		for i in range(len(self.x_batches)):
			batch_x_array = self.x_batches[i]
			lengths = [len(s) for s in batch_x_array]
			
			self.batch_maxlen.append(lengths)
			max_length = max(lengths)
			padding_X = np.zeros([len(batch_x_array), max_length])
			
			for idx,seq in enumerate(batch_x_array):
				padding_X[idx, :len(seq)] = seq
			
			self.x_batches[i] = padding_X
		
	def next_batch(self):
		'''
		get the next batch
		'''
		x = self.x_batches[self.pointer]
		y = self.y_batches[self.pointer]
		s = self.s_batches[self.pointer]
		l = self.l_batches[self.pointer]
		seq_len = self.batch_maxlen[self.pointer]
		self.pointer += 1
		return x, y, s, l, seq_len
		
	def reset_batch_pointer(self):
		'''
		when an epoch is finished, reset the batch pointer to 0
		'''
		self.pointer = 0