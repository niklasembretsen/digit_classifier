import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import pickle
import os


def build_data(set_size = 5000, create_new_data = False):

	if(create_new_data):
		training_set_size = set_size
		D = 10 * training_set_size

		X = np.zeros((D, 1024), dtype=int)
		y = np.zeros(D)
		Y = np.zeros((10, D))

		digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

		for dig in digits:
			print(dig)
			directory = os.fsencode('trainingData/' + dig)

			# train with a limited amount for each digit
			count = 0;

			for file in os.listdir(directory):
				path = os.fsdecode(file)
				image = Image.open( 'trainingData/' + dig + '/' + str(path))
				image = image.convert('L')
				image = image.convert('1')
				image = image.resize((32,32), Image.ANTIALIAS)
				image_array = np.array(image).reshape((1,1024))

				data_point = (int(dig) * training_set_size) + count

				X[data_point] = image_array.astype(int)
				y[data_point] = int(dig)
				Y[int(dig)][data_point] = 1

				count += 1

				if (count == training_set_size):
					break

		df_data = pd.DataFrame([X.T, Y, y])
		df_data.to_pickle('MNIST.pkl')

	df = pd.read_pickle('MNIST-5000.pkl')

	# X = (d x N)
	X = df[0][0].T
	# Y = (K x N)
	Y = df[0][1]
	# y = N x 1
	y = df[0][2]

	return X, Y, y

