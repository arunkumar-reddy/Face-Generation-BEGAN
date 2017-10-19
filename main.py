import sys;
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

import tensorflow as tf;
import argparse;
from model import *;
from dataset import *;

def main(argv):
	parser = argparse.ArgumentParser();
	'''Model Parameters'''
	parser.add_argument('--phase', default='train', help='train or test or interpolate');
	parser.add_argument('--vector_size', default=64, choices=[64,128], help='Latent vector dimensioms');
	parser.add_argument('--num_filters', default=128, choices=[64,128], help='Size of the encoded state in the autoencoder');
	parser.add_argument('--image_shape', default=128, choices=[64,128], help ='Size of the input image to be generated');
	parser.add_argument('--interpolation_module',default='gen',choices=['gen','disc'], help='Module used for interpolation');
	parser.add_argument('--train_dataset', default='/home/arun/Datasets/CelebA/', help='Directory containing the training images');
	parser.add_argument('--train_dir', default='./train/', help='Directory to store training results');
	parser.add_argument('--test_samples', default=30, help='Number of test samples');
	parser.add_argument('--test_dir', default='./test/', help='Directory to store testing images');
	parser.add_argument('--interpolation_dir', default='./interpolate/', help='Directory to store interpolated images');
	parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model');
	parser.add_argument('--save_period', type=int, default=10000, help='Period to save the trained model');
	'''Hyper parameters'''
	parser.add_argument('--solver', default='adam', help='Gradient Descent Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs');
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size');
	parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate');
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay');
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)'); 
	parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)'); 
	parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization');
	parser.add_argument('--gamma', type=float, default=0.4, help='Gamma factor for Boundary equilibrium');
	parser.add_argument('--lambda_k', type=float, default=0.001, help='Proportional gain for the generator loss in discriminator loss');

	args = parser.parse_args();
	with tf.Session() as sess:
		if(args.phase=='train'):
			data = train_data(args);
			model = Model(args,'train');
			sess.run(tf.global_variables_initializer());
			model.Train(sess,data);
		elif(args.phase=='test'):
			model = Model(args,'test');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.Test(sess);
		else:
			data = train_data(args);
			model = Model(args,'interpolate');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.Interpolate(sess,data);

main(sys.argv);