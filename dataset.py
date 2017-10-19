import os;
import math;
import numpy as np;

class Dataset():
	def __init__(self,images,batch_size):
		self.images = np.array(images);
		self.count = len(self.images);
		self.batch_size = batch_size;
		self.batches = int(self.count*1.0/self.batch_size);
		self.index = 0;
		self.indices = list(range(self.count));
		print('Dataset built......');

	def reset(self):
		self.index = 0
		np.random.shuffle(self.indices);

	def next_batch(self):
		if(self.index+self.batch_size<=self.count):
			start = self.index;
			end = self.index+self.batch_size;
			current = self.indices[start:end];
			images = self.images[current];
			self.index += self.batch_size;
			return images;

def train_data(args):
	image_dir = args.train_dataset;
	batch_size = args.batch_size;
	files = os.listdir(image_dir);
	images = [];
	for i in range(0,len(files)):
		images.append(image_dir+files[i]);
	dataset = Dataset(images,batch_size);
	return dataset;