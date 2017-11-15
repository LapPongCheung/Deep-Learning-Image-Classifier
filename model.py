from __future__ import print_function, division 
# from builtins import range

import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt

import tensorflow as tf

from python_speech_features import mfcc 

import sys, os
import random
import time

class CNN(object):
	"""
	Simple CNN model for sound data
	"""
	def __init__(self):
		self.train_file = 'train.txt'
		self.test_file  = 'test.txt'
		self.data_dir = './PNG/'

		self.image_height = 28
		self.image_width = 12
		self.need_shuffle = False

		self.train_list = open(self.train_file, 'rt').read().splitlines()	
		self.test_list = open(self.test_file, 'rt').read().splitlines()		

		if self.need_shuffle:
			random.shuffle(self.train_list)
			random.shuffle(self.test_list)

		self.save_dir = 'model' # save model checkpoints & log message

		# parameters that can be tune
		self.batch_size = 10
		self.max_steps = 1000
		self.log_interval = 5
		self.lr = 1e-3 # learning rate

	def load_png_data(self, phase='TRAIN'):
		# Get full list of images and labels
		if phase == 'TRAIN':
			imglist = [i.split('\t')[0] for i in self.train_list]
			lablist = [int(i.split('\t')[1]) for i in self.train_list]
			num_batch = len(imglist) // self.batch_size
		elif phase == 'TEST':
			imglist = [i.split('\t')[0] for i in self.test_list]
			lablist = [int(i.split('\t')[1]) for i in self.test_list]
			num_batch = len(imglist) // self.batch_size
		else:
			print("W: Invalid phase name")

		timglist = tf.convert_to_tensor(imglist, dtype=tf.string)
		tlablist = tf.convert_to_tensor(lablist, dtype=tf.int32)

		# put list into queue
		data_queue = tf.train.slice_input_producer([timglist, tlablist], 
				capacity=self.batch_size*128, shuffle=False, name=phase)

		# read one image & label
		image_name = tf.string_join([self.data_dir, data_queue[0], '.png'])
		image = tf.image.decode_png(tf.read_file(image_name), channels=1)
		label = data_queue[1]

		# resize image
		image = tf.image.resize_images(image, [self.image_height, self.image_width], 
			method=1, align_corners=True)

		# set date type and shape
		image = tf.cast(image, dtype=tf.float32)
		label = tf.cast(label, dtype=tf.int32)

		# construct a batch of data for training or testing
		dataset = tf.sg_opt()
		dataset.images, dataset.labels = tf.train.batch([image, label], 
				batch_size=self.batch_size, capacity=self.batch_size*128,
				num_threads=5)

		dataset.num_batch = num_batch

		return dataset

	def forward(self, inputs):
		self.network = {}
		self.network["inputs"] = inputs

		reuse = len([t for t in tf.global_variables() if t.name.startswith('CNN')]) > 0	
		with tf.sg_context(name='CNN', size=3, act='relu', reuse=reuse):
			# TODO: Add at least one convolutional layer, pooling layer and fully connected layer 
			predict = None

			return predict

	def train_cnn_with_png(self):
		# load data into dataset
		dataset = self.load_png_data()

		images = dataset.images
		labels = dataset.labels
		num_batch = dataset.num_batch

		# construct network
		self.predict = self.forward(images)

		# define loss function
		self.loss = tf.reduce_mean(self.predict.sg_ce(target=labels))

		# define optimizer
		self.optim = tf.sg_optim(self.loss, optim='Adam', lr=self.lr)

		# define train function
		@tf.sg_train_func
		def alt_train(sess, opt):
			values = sess.run([self.loss, self.optim, self.predict])
			return values[0]

		# execute train function
		tic = time.time()
		alt_train(log_interval=self.log_interval, ep_size=num_batch, max_ep=self.max_steps,
			early_stop=True, save_dir=self.save_dir)
		toc = time.time()

		print("I: Training time = {} sec".format(toc-tic))

	def test_cnn_with_png(self):
		dataset = self.load_png_data(phase='TEST')

		images = dataset.images
		labels = dataset.labels
		num_batch = dataset.num_batch

		predict = self.forward(images)

		# initialize tf session
		sess = tf.Session()
		tf.sg_init(sess)

		# restore pre-trained model
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

		# start queue runner
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		total_example = (num_batch*self.batch_size)
		correct_num = 0
		for i in range(num_batch):
			predict_value, gt = sess.run([tf.argmax(predict, axis=1), labels])
			tmp = predict_value - gt
			correct_num += np.count_nonzero(tmp == 0)

		accuracy = correct_num / total_example
		print("Evaluate {} example, accuracy = {} %".format(total_example, accuracy*100.0))

	def demo(self, name='9_08.wav'):
		# extract mfcc feature from wav file
		(rate,sig) = wav.read(name)
		mfcc_feat = mfcc(sig,rate, nfft=2048, numcep=13)
		mfcc_featn = mfcc_feat[:, 1:13]

		# save the feature as png image
		imsave("mfcc_features.png", mfcc_featn)

		# restore the feature
		mfcc_png = imread("mfcc_features.png", mode='L')
		img = imresize(mfcc_png, [self.image_height, self.image_width], interp='bicubic')
		img = img[np.newaxis, :,:, np.newaxis]

		# input placeholder for neural network
		inputs_tensor = tf.placeholder(shape=img.shape, dtype=tf.float32)

		# restore network structure
		predict = self.forward(inputs_tensor)

		sess = tf.Session()
		tf.sg_init(sess)

		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

		predict_value = sess.run(tf.argmax(predict, axis=1), 
			feed_dict = {inputs_tensor: img})[0]

		print("Number {}".format(predict_value))
