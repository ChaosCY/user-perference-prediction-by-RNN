import tensorflow as tf
import numpy as np

class BASIC_LSTM():
	'''
	basic LSTM model
	'''
	def __init__(self, args):
		self.args = args
		
		num_durations = args.num_durations
		num_risks = args.num_risks
		num_returns = args.num_returns
		num_amounts = args.num_amounts
		num_locations = args.num_locations
		
		# input datas
		self.input_duration = tf.placeholder(tf.int32, shape=[None, None], name='input_duration')
		self.input_risk = tf.placeholder(tf.int32, shape=[None, None], name='input_risk')
		self.input_return = tf.placeholder(tf.int32, shape=[None, None], name='input_return')
		self.input_amount = tf.placeholder(tf.int32, shape=[None, None], name='input_amount')
		self.input_location = tf.placeholder(tf.int32, shape=[None, None], name='input_location')
		self.targets_duration = tf.placeholder(tf.int32, shape=[None], name='targets_duration')
		self.targets_risk = tf.placeholder(tf.int32, shape=[None], name='targets_risk')
		self.targets_return = tf.placeholder(tf.int32, shape=[None], name='targets_return')
		self.targets_amount = tf.placeholder(tf.int32, shape=[None], name='targets_amount')
		self.targets_location = tf.placeholder(tf.int32, shape=[None], name='targets_location')
		self.last_duration = tf.placeholder(tf.int32, shape=[None], name='last_duration')
		self.last_risk = tf.placeholder(tf.int32, shape=[None], name='last_risk')
		self.last_return = tf.placeholder(tf.int32, shape=[None], name='last_return')
		self.last_amount = tf.placeholder(tf.int32, shape=[None], name='last_amount')
		self.last_location = tf.placeholder(tf.int32, shape=[None], name='last_location')
		
		# onehot encode
		y_duration = tf.one_hot(self.targets_duration, num_durations)
		y_risk = tf.one_hot(self.targets_risk, num_risks)
		y_return = tf.one_hot(self.targets_return, num_returns)
		y_amount = tf.one_hot(self.targets_amount, num_amounts)
		y_location = tf.one_hot(self.targets_location, num_locations)
		last_duration = tf.one_hot(self.last_duration, num_durations)
		last_risk = tf.one_hot(self.last_risk, num_risks)
		last_return = tf.one_hot(self.last_return, num_returns)
		last_amount = tf.one_hot(self.last_amount, num_amounts)
		last_location = tf.one_hot(self.last_location, num_locations)
		
		self.switch = tf.placeholder(tf.float32, shape=[None, None], name='switch')
		self.seq_len = tf.placeholder(tf.int32, [None])
		self.keep_prob = tf.placeholder(tf.float32)
		
		loss_duration, self.seq_duration = self.model_graph(args, 'duration', num_durations, self.input_duration, y_duration, last_duration)
		loss_risk, seq_risk = self.model_graph(args, 'risk', num_risks, self.input_risk, y_risk, last_risk)
		loss_return, seq_return = self.model_graph(args, 'return', num_returns, self.input_return, y_return, last_return)
		loss_amount, seq_amount = self.model_graph(args, 'amount', num_amounts, self.input_amount, y_amount, last_amount)
		loss_location, seq_location = self.model_graph(args, 'location', num_locations, self.input_location, y_location, last_location)
		
		loss_sum = tf.add_n([loss_duration, loss_risk, loss_return, loss_amount, loss_location])
		
		seq_duration = tf.expand_dims(self.seq_duration, -1)
		seq_risk = tf.expand_dims(seq_risk, -1)
		seq_return = tf.expand_dims(seq_return, -1)
		seq_amount = tf.expand_dims(seq_amount, -1)
		seq_location = tf.expand_dims(seq_location, -1)
		self.predict_sequence = tf.concat([seq_duration, seq_risk, seq_return, seq_amount, seq_location], 1)
		
		self.loss = tf.reduce_sum(loss_sum)
		
		self.lr = tf.Variable(0.0, trainable=False)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		
	def model_graph(self, args, scope_name, num_items, input_data, y_, last_):
		with tf.variable_scope(scope_name):
			rnn_fw = tf.contrib.rnn.BasicLSTMCell(args.num_factors)
			#rnn_fw = tf.contrib.rnn.DropoutWrapper(rnn_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
			
			softmax_w = tf.get_variable("softmax_w", [10, num_items], initializer=tf.contrib.layers.xavier_initializer())
			softmax_b = tf.get_variable("softmax_b", [num_items], initializer=tf.constant_initializer(0.0))
			switch_w = tf.get_variable("switch_w", [10, num_items*num_items], initializer=tf.contrib.layers.xavier_initializer())
			switch_b = tf.get_variable("switch_b", [num_items*num_items], initializer=tf.contrib.layers.xavier_initializer())
			
			embedding = tf.get_variable("lable_embedding", [num_items, args.num_factors])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
			
			_, state = tf.nn.dynamic_rnn(
				rnn_fw,
				inputs,
				dtype=tf.float32,
				sequence_length=self.seq_len
			)
		
			state = tf.concat([state[0], state[1]], 1)
			
			s_gate = tf.nn.relu(tf.matmul(self.switch, switch_w) + switch_b)
			s_gate = tf.reshape(s_gate, [-1, num_items, num_items])
			last_label = tf.expand_dims(last_, 1)
			label_s_dis = tf.reshape(tf.matmul(last_label, s_gate), [-1, num_items])
			logits = tf.matmul(state, softmax_w) + softmax_b
			logits_new = tf.add_n([logits, label_s_dis])
			y = tf.nn.softmax(logits)
			
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
			
			predict_sequence = tf.argmax(y, axis=1)
			
			return loss, predict_sequence