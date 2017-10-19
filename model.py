import os;
import sys;
import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt;

from tqdm import tqdm;
from skimage.io import imread,imsave,imshow;
from skimage.transform import resize;
from dataset import *;
from generator import *;
from autoencoder import *;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.vector_size = params.vector_size;
		self.num_filters = params.num_filters;
		self.batch_size = 1 if phase=='test' else params.batch_size;
		self.image_shape = [params.image_shape,params.image_shape,3];
		self.gamma = params.gamma;
		self.lambda_k = params.lambda_k; 
		self.save_dir = os.path.join(params.save_dir,self.params.solver+'/');
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.learning_rate = tf.Variable(params.learning_rate,name='learning_rate');
		self.learning_rate_update = tf.assign(self.learning_rate,tf.maximum(self.learning_rate*0.5,0.00002),name='lr_update');
		self.saver = tf.train.Saver(max_to_keep = 100);
		self.build();

	def build(self):
		print('Building the Model......');
		image_shape = self.image_shape;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		vectors = tf.placeholder(tf.float32,[self.batch_size,self.vector_size]);
		train = tf.placeholder(tf.bool);
		reuse = False if self.phase =='train' else True;
		k = tf.Variable(0.,name='k_t',trainable=False);
		generator = Generator(self.params,self.phase);
		autoencoder = Autoencoder(self.params,self.phase);
		output = generator.run(vectors,train,reuse);
		real = autoencoder.run(images,train,reuse);
		fake = autoencoder.run(output,train,reuse=True);
		real_loss = tf.reduce_mean(tf.abs(real-images));
		fake_loss = tf.reduce_mean(tf.abs(fake-output));
		gen_loss = tf.reduce_mean(tf.abs(fake-output));
		disc_loss = real_loss-k*fake_loss;
		balance = self.gamma*real_loss-gen_loss;
		measure = real_loss+tf.abs(balance);
		
		self.images = images;
		self.vectors = vectors;
		self.train = train;
		self.generator = generator;
		self.autoencoder = autoencoder;
		self.disc_loss = disc_loss;
		self.gen_loss = gen_loss;
		self.real_loss = real_loss;
		self.fake_loss = fake_loss;
		self.output = output;
		self.balance = balance;
		self.measure = measure;
		self.k = k;

		if self.params.solver == 'adam':
			disc_solver = tf.train.AdamOptimizer(self.learning_rate);
			gen_solver = tf.train.AdamOptimizer(self.learning_rate);
		elif self.params.solver == 'momentum':
			disc_solver = tf.train.MomentumOptimizer(self.learning_rate,self.params.momentum);
			gen_solver = tf.train.MomentumOptimizer(self.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			disc_solver = tf.train.RMSPropOptimizer(self.learning_rate,self.params.weight_decay,self.params.momentum);
			gen_solver = tf.train.RMSPropOptimizer(self.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			disc_solver = tf.train.GradientDescentOptimizer(self.learning_rate);
			gen_solver = tf.train.GradientDescentOptimizer(self.learning_rate);

		tensorflow_variables = tf.trainable_variables();
		disc_variables = [variable for variable in tensorflow_variables if 'd_' or 'e_' in variable.name];
		gen_variables = [variable for variable in tensorflow_variables if 'g_' in variable.name];
		disc_gradients,_ = tf.clip_by_global_norm(tf.gradients(self.disc_loss,disc_variables),3.0);
		gen_gradients,_ = tf.clip_by_global_norm(tf.gradients(self.gen_loss,gen_variables),3.0);
		disc_optimizer = disc_solver.apply_gradients(zip(disc_gradients,disc_variables));
		gen_optimizer = gen_solver.apply_gradients(zip(gen_gradients,gen_variables),global_step=self.global_step);
		self.k = tf.assign(self.k,tf.clip_by_value(self.k+self.lambda_k*self.balance,0,1));
		self.disc_optimizer = disc_optimizer;
		self.gen_optimizer = gen_optimizer;

		if(self.phase=='interpolate'):
			interpolation_vectors = tf.get_variable('z',[self.batch_size,self.vector_size],tf.float32,tf.random_uniform_initializer(-1,1));
			interpolation_output = generator.run(interpolation_vectors,train,reuse=True);
			interpolation_loss = tf.reduce_mean(tf.abs(images-interpolation_output));
			interpolation_solver = tf.train.AdamOptimizer(0.0001);
			interpolation_optimizer = interpolation_solver.minimize(interpolation_loss,var_list=[interpolation_vectors]);

			self.interpolation_vectors = interpolation_vectors;
			self.interpolation_loss = interpolation_loss;
			self.interpolation_optimizer = interpolation_optimizer;

		print('Model built......');

	def Train(self,sess,data):
		print('Training the Model......');
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(data.batches)),desc='Batch'):
				files = data.next_batch();
				images = self.load_images(files);
				vectors = np.random.uniform(-1,1,size=(self.batch_size,self.vector_size));
				feed_dict = {self.images:images,self.vectors:vectors,self.train:True};
				global_step,disc_loss,gen_loss,measure,k,_,_ = sess.run([self.global_step,self.disc_loss,self.gen_loss,self.measure,self.k,self.disc_optimizer,self.gen_optimizer],feed_dict=feed_dict);
				print(' Convergence = %f Autoencoder_loss = %f Generator_loss = %f'%(measure,disc_loss,gen_loss));
				if(global_step%5000==0):
					output = sess.run(self.output,feed_dict={self.vectors:vectors, self.train:False});
					self.save_image(output[0],'train_sample_'+str(global_step));
				if(global_step%100000==0):
					sess.run(self.learning_rate_update);
				if(global_step%self.params.save_period==0):
					self.save(sess);
			data.reset();
		self.save(sess);
		print('Model trained......');

	def Test(self,sess):
		print('Testing the Model......');
		for i in tqdm(list(range(self.params.test_samples)),desc='Batch'):
			vector = np.random.uniform(-1,1,size=(self.vector_size));
			output = sess.run(self.output,feed_dict={self.vectors:vector, self.train:False});
			self.save_image(output,'test_sample_'+str(i+1));
		print('Testing completed......');

	def Interpolate(self,sess,data):
		print('Interpolation......');
		batch_size = self.batch_size;
		for index in range(3):
			files = data.next_batch();
			images = self.load_images(files);
			if(self.params.interpolation_module=='gen'):
				for i in range(100):
					loss,_ = sess.run([self.interpolation_loss,self.interpolation_optimizer],feed_dict={self.images:images, self.train:False});
				vectors = sess.run(self.interpolation_vectors);
				vectors1,vectors2 = vectors[:batch_size/2],vectors[batch_size/2:];
				ratios = np.linspace(0,1,10);
				for i in range(0,10,2):
					v1 = np.stack([slerp(ratio[i],r1,r2) for r1,r2 in zip(vectors1,vectors2)]);
					v2 = np.stack([slerp(ratio[i+1],r1,r2) for r1,r2 in zip(vectors1,vectors2)]);
					vectors = np.concatenate(v1,v2);
					output = sess.run(self.output,feed_dict={self.vectors:vectors, self.train:False});
					for j in range(len(output)):
						self.save_image(output[j],'interpG_{}_{}'.format(index,j));
			else:
				vectors = sess.run(self.autoencoder.vectors,feed_dict={self.images:images, self.train:False});
				vectors1,vectors2 = vectors[:batch_size/2],vectors[batch_size/2:];
				ratios = np.linspace(0,1,10);
				for i in range(0,10,2):
					v1 = np.stack([slerp(ratio[i],r1,r2) for r1,r2 in zip(vectors1,vectors2)]);
					v2 = np.stack([slerp(ratio[i+1],r1,r2) for r1,r2 in zip(vectors1,vectors2)]);
					vectors = np.concatenate(v1,v2);
					output = sess.run(self.autoencoder.outputs,feed_dict={self.autoencoder.vectors:vectors});
					for j in range(len(output)):
						self.save_image(output[j],'interpD_{}_{}'.format(index,j));
		print('Interpolation completed......');

	def slerp(self,val,low,high):
		omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low),high/np.linalg.norm(high)),-1,1));
		theta = np.sin(omega);
		if(theta==0):
			return (1.0-val)*low+val*high # L'Hopital's rule/LERP
		return np.sin((1.0-val)*omega)/theta*low + np.sin(val*omega)/theta*high;
	
	def save(self,sess):
		print(('Saving model to %s......'% self.save_dir));
		self.saver.save(sess,self.save_dir,self.generator_step);

	def load(self,sess):
		print('Loading model.....');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first...");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def load_images(self,files):
		images = [];
		image_shape = self.image_shape;
		for image_file in files:
			image = imread(image_file);
			image = resize(image,(image_shape[0],image_shape[1]));
			image = (image-127.5)/127.5;
			images.append(image);
		images = np.array(images,np.float32);
		return images;

	def save_image(self,output,name):
		output = (output*127.5)+127.5;
		if(self.phase=='train'):
			file_name = self.params.train_dir+name+'.png';
		elif(self.phase=='test'):
			file_name = self.params.test_dir+name+'.png';
		else:
			file_name = self.params.interpolation_dir+name+'.png';
		imsave(file_name,output);
		print('Saving the image %s...',file_name);