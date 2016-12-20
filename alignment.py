__author__ = 'Alexey'

import json
from pylab import *
from scipy import optimize
import numpy as np
def triangle(noise_level, half_width, height, x0):
    def f(x):
        d = abs(x-x0)
        if d>half_width :
            return noise_level
        return noise_level + (1-abs(x-x0)/half_width)*height
    return np.vectorize(f)

def fittriangle(data):
    params = [np.mean(data[1]), 2000.0, np.max(data[1]),np.mean(data[0])]
    errorf = lambda p: ravel(triangle(*p)(data[0])-data[1])
    p,success = optimize.leastsq(errorf, params, maxfev = 1000)
    return p

fname = '25-06-2015_11-19-43.json'
data_file = open(fname,'r')
data = json.load(data_file)
data = data["data"]

counts = []
shift = []
for row in data:
    shift.append(row[3])
    counts.append(row[-1])

params = fittriangle([shift,counts])
#params = [10.0, 2000.0, 20.0,0.0]
print(params)
fitF = triangle(*params)
fitplot_shifts = np.arange(shift[0],shift[-1],(shift[-1]-shift[0])/1000)
tr_counts = fitF(fitplot_shifts)


ax = plt.subplot(111)
ax.plot(fitplot_shifts,tr_counts,'k-',label='Shift={0:.0f}\nHWidth={1:.0f}'.format(params[3],params[1]))
ax.plot(shift, counts,'r+')
ax.legend(loc=4)
plt.xlabel('Shift, ns')
plt.ylabel('Counts')
plt.show()