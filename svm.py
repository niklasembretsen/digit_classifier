import numpy as np
import pylab as pl
import math
import cvxopt
import dataReader as dr
import pickle
import sys

from cvxopt.solvers import qp
from cvxopt.base import matrix

slack = False

def linearKernel(x, y):
	res = np.dot(x,y) + 1
	return res

def polyKernel(x, y, polyDegree = 2):
	res = (np.dot(x[0],y[0]) + 1)**polyDegree
	return res

def rbfKernel(x, y, sigma = 0.1):
	x = np.array(x)
	y = np.array(y)
	res = np.exp(-(np.linalg.norm(x-y)**2)/2*(sigma**2))
	return res

def sigmoidKernel(x, y, k = 1, delta = 0):
	res = np.dot(x, y)
	return np.tanh(res - delta)


def buildPMatrix(data, size):
	pMatrix = []
	for dataPoint in data:
		rowList = [];
		point = [dataPoint[:-1]]
		t = dataPoint[-1]
		for points in data:		
			point2 = [points[:-1]]
			t2 = points[-1]
			tempVal = t*t2*polyKernel(point, point2)
			rowList.append(float(tempVal))
		pMatrix.append(rowList)

	return matrix(pMatrix, (size, size), 'd')

def buildGMatrix(size):
	if slack == True:
		gMatrix = []
		for i in range(0,size*2):
			tempRow = []
			for j in range(0,size):
				if i <= 19:
					if i == j:
						tempRow.append(-1)
					else:
						tempRow.append(0)
				else:
					if (i - size) == j:
						tempRow.append(1)
					else:
						tempRow.append(0)

			gMatrix.append(tempRow)

		gMatrix = np.array(gMatrix)

		return matrix(gMatrix, (2*size, size), 'd')
	else:
		gMatrix = []
		for i in range(0,size):
			tempRow = []
			for j in range(0,size):
				if i == j:
					tempRow.append(-1)
				else:
					tempRow.append(0)

			gMatrix.append(tempRow)

		gMatrix = np.array(gMatrix)

		return matrix(gMatrix, (size, size), 'd')


def buildQVector(size):
	q = []
	for i in range(0, size):
		q.append(-1)

	return matrix(q, (size, 1), 'd')

def buildHVector(size):
	h = []
	for i in range(0, size):
		h.append(0)

	if slack == True:
		for i in range(0, size):
			h.append(1)

		return matrix(h, (2*size, 1), 'd')
	else:
		return matrix(h, (size, 1), 'd')

def getAlphaValues(data):
	size = len(data)
	P = buildPMatrix(data, size).trans()
	G = buildGMatrix(size)
	q = buildQVector(size)
	h = buildHVector(size)
	r = qp(P, q, G, h)
	return r

def findOptimalSolution(digit, trainingSetSize):

	data = dr.buildData(digit, trainingSetSize)
	r = getAlphaValues(data);
	alpha = list(r['x'])

	return [alpha, data]

def trimAlpha(alphaAndData):
	alphas = alphaAndData[0]
	data = alphaAndData[1]

	nonZeroAlphas = []
	for i, alpha in enumerate(alphas):
		if(alpha > 1.0e-5):
			nonZeroAlphas.append([alpha, data[i]])

	return nonZeroAlphas

def indicator(point, nonZeroAlphas):

	indX = 0
	alphaAndData = nonZeroAlphas
	alpha = alphaAndData[0]
	dataPoint = alphaAndData[1]
	# print(alpha)
	# print(dataPoint[:-1])

	indX += alpha * dataPoint[-1] * polyKernel(point, dataPoint[:-1])

	return indX

def trainModel(trainingSetSize):
	digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	for dig in digits:
		alphaAndData = findOptimalSolution(dig, trainingSetSize)
		nonZeroAlphas = trimAlpha(alphaAndData)
		pickle.dump(nonZeroAlphas, open('models/' + dig + '.p', 'wb'))
