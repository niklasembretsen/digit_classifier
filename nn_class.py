import numpy as np
import pickle
import pandas as pd

"Calculate softmax"
# returns:
#	normalized exponential values of vector s
def softmax(s):
	return np.exp(s)/np.sum(np.exp(s), axis=0)

"Perform a forward pass"
# evaluates the network function,
# X = the dataset
# theta = [W_1, b_1, W_2, b_2]
# ----------------------------
# s_1 = W_1X + b_1 (m x N)
# h = max(0, s_1) (ReLu activation function)
# s = W_2h + b2 = ([K x m] x [m x N] => K x N)
# i.e softmax(W_2 x max(0, [WX + b]))
# returns:
#	P - (K x N) matrix of the probabilities for each class
# 	s_1, h for gradient computations
def forward_pass(X):

	df = pd.read_pickle('params.pkl')
	theta = [df[0][0], df[0][1], df[0][2], df[0][3]]

	b_1 = theta[1].reshape((len(theta[1]),))
	b_2 = theta[3].reshape((len(theta[3]),))

	print("W_1", theta[0].shape)
	print("W_2", theta[2].shape)
	print("b_1", b_1.shape)
	print("b_2", b_2.shape)

	s_1 = np.matmul(theta[0], X)
	s_1 = np.add(s_1, b_1)
	print("s_1", s_1.shape)
	h = np.maximum(0, s_1)
	print("h", h.shape)
	s = np.matmul(theta[2], h)
	print("s", s.shape)
	s = np.add(s, b_2)
	P = softmax(s)

	return P