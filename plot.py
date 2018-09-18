import numpy as np
import svm 
import pylab
import randomData as rd

from cvxopt.base import matrix

xrange = np.arange(-4, 4, 0.05)
yrange = np.arange(-4, 4, 0.05)

alphas = svm.findOptimalSolution()
nonZeroAlphas = svm.trimAlpha(alphas)

grid = matrix([[svm.indicator(x, y, nonZeroAlphas)
	for y in yrange]
	for x in xrange])

pylab.plot([n[1][0] for n in nonZeroAlphas], [n[1][1] for n in nonZeroAlphas], 'go')

pylab.contour(xrange, yrange, grid,
	(-1.0, 0.0, 1.0),
	colors=('red', 'black', 'blue'),
	linewidths=(1, 3, 1))

pylab.show()