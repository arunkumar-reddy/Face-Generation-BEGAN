import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class Autoencoder(object):
	def __init__(self,params,phase):
		self.phase = phase;
		self.image_shape = params.image_shape;
		self.vector_size = params.vector_size;
		self.num_filters = params.num_filters;
		self.batch_size = 1 if phase=='test' else params.batch_size;
		self.batch_norm = params.batch_norm;
		print('Building the Autoencoder......');

	def run(self,images,train,reuse=False):
		vectors = self.encode(images,train,reuse);
		output = self.decode(vectors,train,reuse);
		return output;

	def encode(self,images,train,reuse=False):
		bn = self.batch_norm;
		num_filters = self.num_filters;
		with tf.variable_scope('encoder') as scope:
			if reuse:
				scope.reuse_variables();
			conv1_1 = convolution(images,3,3,num_filters,1,1,'e_conv1_1');
			conv1_1 = batch_norm(conv1_1,'e_bn1_1',train,bn,'elu');
			conv1_2 = convolution(conv1_1,3,3,num_filters,1,1,'e_conv1_2');
			conv1_2 = batch_norm(conv1_2,'e_bn1_2',train,bn,'elu');
			conv1_3 = convolution(conv1_2,3,3,num_filters,1,1,'e_conv1_3');
			conv1_3 = batch_norm(conv1_3,'e_bn1_3',train,bn,'elu');
			conv1_4 = convolution(conv1_3,3,3,num_filters,2,2,'e_conv1_4');
			conv1_4 = batch_norm(conv1_4,'e_bn1_4',train,bn,'elu');

			conv2_1 = convolution(conv1_4,3,3,num_filters*2,1,1,'e_conv2_1');
			conv2_1 = batch_norm(conv2_1,'e_bn2_1',train,bn,'elu');
			conv2_2 = convolution(conv2_1,3,3,num_filters*2,1,1,'e_conv2_2');
			conv2_2 = batch_norm(conv2_2,'e_bn2_2',train,bn,'elu');
			conv2_3 = convolution(conv2_2,3,3,num_filters*2,2,2,'e_conv2_3');
			conv2_3 = batch_norm(conv2_3,'e_bn2_3',train,bn,'elu');

			conv3_1 = convolution(conv2_3,3,3,num_filters*4,1,1,'e_conv3_1');
			conv3_1 = batch_norm(conv3_1,'e_bn3_1',train,bn,'elu');
			conv3_2 = convolution(conv3_1,3,3,num_filters*4,1,1,'e_conv3_2');
			conv3_2 = batch_norm(conv3_2,'e_bn3_2',train,bn,'elu');
			conv3_3 = convolution(conv3_2,3,3,num_filters*4,2,2,'e_conv3_3');
			conv3_3 = batch_norm(conv3_3,'e_bn3_3',train,bn,'elu');			

			conv4_1 = convolution(conv3_3,3,3,num_filters*4,1,1,'e_conv4_1');
			conv4_1 = batch_norm(conv4_1,'e_bn4_1',train,bn,'elu');
			conv4_2 = convolution(conv4_1,3,3,num_filters*4,1,1,'e_conv4_2');
			conv4_2 = batch_norm(conv4_2,'e_bn4_2',train,bn,'elu');

			if(self.image_shape==64):
				feats = tf.reshape(conv4_2,[self.batch_size,8*8*num_filters*4]);
				feats = tf.fully_connected(feats,self.vector_size,'e_fc1');
				self.vectors = feats;
			else:
				conv4_3 = convolution(conv4_2,3,3,num_filters*4,2,2,'e_conv4_3');
				conv4_3 = batch_norm(conv4_3,'e_bn4_3',train,bn,'elu');
				conv5_1 = convolution(conv4_3,3,3,num_filters*4,1,1,'e_conv5_1');
				conv5_1 = batch_norm(conv5_1,'e_bn5_1',train,bn,'elu');
				conv5_2 = convolution(conv5_1,3,3,num_filters*4,1,1,'e_conv5_2');
				conv5_2 = batch_norm(conv5_2,'e_bn5_2',train,bn,'elu');
				feats = tf.reshape(conv5_2,[self.batch_size,8*8*num_filters*4]);
				feats = fully_connected(feats,self.vector_size,'e_fc1');
				self.vectors = feats;
			
			return self.vectors;

	def decode(self,vectors,train,reuse=False):
		bn = self.batch_norm;
		num_filters = self.num_filters;
		with tf.variable_scope('decoder') as scope:
			if reuse:
				scope.reuse_variables();
			feats = fully_connected(vectors,8*8*num_filters,'d_fc1');
			feats = tf.reshape(feats,[self.batch_size,8,8,num_filters]);

			conv1_1 = convolution(feats,3,3,num_filters,1,1,'d_conv1_1');
			conv1_1 = batch_norm(conv1_1,'d_bn1_1',train,bn,'elu');
			conv1_2 = convolution(conv1_1,3,3,num_filters,1,1,'d_conv1_2');
			conv1_2 = batch_norm(conv1_2,'d_bn1_2',train,bn,'elu');
			conv1_2 = upscale(conv1_2,2);

			conv2_1 = convolution(conv1_2,3,3,num_filters,1,1,'d_conv2_1');
			conv2_1 = batch_norm(conv2_1,'d_bn2_1',train,bn,'elu');
			conv2_2 = convolution(conv2_1,3,3,num_filters,1,1,'d_conv2_2');
			conv2_2 = batch_norm(conv2_2,'d_bn2_2',train,bn,'elu');
			conv2_2 = upscale(conv2_2,2);

			conv3_1 = convolution(conv2_2,3,3,num_filters,1,1,'d_conv3_1');
			conv3_1 = batch_norm(conv3_1,'d_bn3_1',train,bn,'elu');
			conv3_2 = convolution(conv3_1,3,3,num_filters,1,1,'d_conv3_2');
			conv3_2 = batch_norm(conv3_2,'d_bn3_2',train,bn,'elu');
			conv3_2 = upscale(conv3_2,2);

			conv4_1 = convolution(conv3_2,3,3,num_filters,1,1,'d_conv4_1');
			conv4_1 = batch_norm(conv4_1,'d_bn4_1',train,bn,'elu');
			conv4_2 = convolution(conv4_1,3,3,num_filters,1,1,'d_conv4_2');
			conv4_2 = batch_norm(conv4_2,'d_bn4_2',train,bn,'elu');

			if(self.image_shape==64):
				conv4_3 = convolution(conv4_2,3,3,3,1,1,'d_conv4_3');
				self.output = conv4_3;
			else:
				conv4_2 = upscale(conv4_2,2);
				conv5_1 = convolution(conv4_2,3,3,num_filters,1,1,'d_conv5_1');
				conv5_1 = batch_norm(conv5_1,'d_bn5_1',train,bn,'elu');
				conv5_2 = convolution(conv5_1,3,3,num_filters,1,1,'d_conv5_2');
				conv5_2 = batch_norm(conv5_2,'d_bn5_2',train,bn,'elu');
				conv5_3 = convolution(conv5_2,3,3,3,1,1,'d_conv5_3');
				self.output = conv5_3;

			return nonlinear(self.output,'tanh');