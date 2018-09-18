import pylab as pl
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import os


def buildData():

	image = Image.open('sample/img0.png')
	image = image.convert('L')
	image = image.convert('1')
	image = image.resize((32,32), Image.ANTIALIAS)
	image_array = np.array(image)

	data = []
	for row in image_array:
		for pix in row:
			data.append(int(pix))
			print(int(pix), end="")
		print()

	return np.array(data)
