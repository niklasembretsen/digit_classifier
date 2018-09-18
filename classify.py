import svm 
import inputData as inD 
import pickle
import sys

def classifyImg(data):

	digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	score = []
	for dig in digits:
		nonZeroAlphas = pickle.load(open('models/' + dig + '.p', 'rb'))
		tempScore = 0
		for alphasAndData in nonZeroAlphas:
			tempScore += svm.indicator(data, alphasAndData)
		score.append(tempScore)

	tempMax = 0
	index = 0
	for sc in score:
		print(sc)

	for idx, val in enumerate(score):
		if(val > tempMax):
			tempMax = val
			index = idx

	print("Digit most probable is: ", index)

data = inD.buildData()
classifyImg(data)