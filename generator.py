import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class Generator(object):
	def __init__(self,params,phase):
		self.phase = phase;
		self.image_shape = params.image_shape;
		self.vector_size = params.vector_size;
		self.num_filters = params.num_filters;
		self.batch_size = 1 if phase=='test' else params.batch_size;
		self.batch_norm = params.batch_norm;
		print('Building the Generator......');

	def run(self,vectors,train,reuse=False):
		bn = self.batch_norm;
		num_filters = self.num_filters;
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables();
			feats = fully_connected(vectors,8*8*num_filters,'g_fc1');
			feats = tf.reshape(feats,[self.batch_size,8,8,num_filters]);

			conv1_1 = convolution(feats,3,3,num_filters,1,1,'g_conv1_1');
			conv1_1 = batch_norm(conv1_1,'g_bn1_1',train,bn,'elu');
			conv1_2 = convolution(conv1_1,3,3,num_filters,1,1,'g_conv1_2');
			conv1_2 = batch_norm(conv1_2,'g_bn1_2',train,bn,'elu');
			conv1_2 = upscale(conv1_2,2);

			conv2_1 = convolution(conv1_2,3,3,num_filters,1,1,'g_conv2_1');
			conv2_1 = batch_norm(conv2_1,'g_bn2_1',train,bn,'elu');
			conv2_2 = convolution(conv2_1,3,3,num_filters,1,1,'g_conv2_2');
			conv2_2 = batch_norm(conv2_2,'g_bn2_2',train,bn,'elu');
			conv2_2 = upscale(conv2_2,2);

			conv3_1 = convolution(conv2_2,3,3,num_filters,1,1,'g_conv3_1');
			conv3_1 = batch_norm(conv3_1,'g_bn3_1',train,bn,'elu');
			conv3_2 = convolution(conv3_1,3,3,num_filters,1,1,'g_conv3_2');
			conv3_2 = batch_norm(conv3_2,'g_bn3_2',train,bn,'elu');
			conv3_2 = upscale(conv3_2,2);

			conv4_1 = convolution(conv3_2,3,3,num_filters,1,1,'g_conv4_1');
			conv4_1 = batch_norm(conv4_1,'g_bn4_1',train,bn,'elu');
			conv4_2 = convolution(conv4_1,3,3,num_filters,1,1,'g_conv4_2');
			conv4_2 = batch_norm(conv4_2,'g_bn4_2',train,bn,'elu');

			if(self.image_shape==64):
				conv4_3 = convolution(conv4_2,3,3,3,1,1,'g_conv4_3');
				self.output = conv4_3;
			else:
				conv4_2 = upscale(conv4_2,2);
				conv5_1 = convolution(conv4_2,3,3,num_filters,1,1,'g_conv5_1');
				conv5_1 = batch_norm(conv5_1,'g_bn5_1',train,bn,'elu');
				conv5_2 = convolution(conv5_1,3,3,num_filters,1,1,'g_conv5_2');
				conv5_2 = batch_norm(conv5_2,'g_bn5_2',train,bn,'elu');
				conv5_3 = convolution(conv5_2,3,3,3,1,1,'g_conv5_3');
				self.output = conv5_3;

			self.vectors = vectors;	
			return self.output;