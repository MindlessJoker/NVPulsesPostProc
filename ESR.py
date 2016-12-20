__author__ = 'Vadim'

import json
from matplotlib import pyplot as plt
import numpy as np
import os
from pylab import *
from scipy import optimize

def lorentzian(position, width, alpha):

    return lambda freq: 1 - (width*alpha)/((freq-position)**2+(width**2)/4)

def lorentzian2(position1,position2, width1, width2, alpha1, alpha2):

    return lambda freq: 1 - (width1*alpha1)/((freq-position1)**2+(width1**2)/4) - (width2*alpha2)/((freq-position2)**2+(width2**2)/4)



def fitlorentzian(data):
    params = [2880, 2840, 5, 5, 0.3, 0.2]
    errorf = lambda p: ravel(lorentzian2(*p)(data[0])-data[1])
    p,success = optimize.leastsq(errorf, params, maxfev = 1000)
    return p

class esr_data():
    def __init__(self, filename):
        self.filename = filename
        self.filehead, self.filetail = os.path.split(self.filename)
        self.file_parsing()
        #self.plot_data()
    def plot_data(self, ax):
        freq = []
        counts = []
        try:
            for row in self.data:
                freq.append(row[0])
                counts.append(row[-1])
        except:
            print('Oops, data file does not exist')
        ax.plot(freq, counts, 'o-',label = self.filetail[0:-5])
        plt.legend()


        #plt.show()
    def plot_normalized_data(self,ax, fit = False, legend = False,loc=4, plot_id=0):
        formats = ['*-','o-','v-','+-']
        freq = []
        counts = []
        try:
            for row in self.data:
                freq.append(row[0])
                counts.append(row[-1]/row[-2])
        except:
            print('Oops, data file does not exist')


        #plt.show()
        if fit:
            self.params = fitlorentzian([freq,counts])
            #fitF = lorentzian2(*self.params)
            fitF = lorentzian2(2890,2840,10,10,0.3,0.3)
            print(self.params)
            fitplotfreq = np.arange(freq[0],freq[-1],(freq[-1]-freq[0])/1000.0)
            #print(fitplotfreq)
            fitdata = fitF(fitplotfreq)
            #fitdata = np.zeros(fitplotfreq.shape)
            print('fit!')
            label = 'Lorentz fit\n-Width = '\
                    +("%.1f" % float(self.params[2]))\
                    + 'Mhz'+',\n-Position ='\
                    +("%.3f" % (float(self.params[0])/1000))+' GHz'\
                    +("%.3f" % (float(self.params[1])/1000))+' GHz'
            ax.plot(fitplotfreq,fitdata,'k-',lw=4, label = label)
            ax.plot(freq, counts,formats[plot_id%len(formats)], label = self.filetail[0:-5])
        else:
            ax.plot(freq, counts,formats[plot_id%len(formats)], label = self.filetail[0:-5])
        if legend:
            plt.legend(loc=loc)

    def file_parsing(self):
        data_file = open(self.filename,'r')
        data = json.load(data_file)
        #print(data["data"][0])
        self.data = data["data"]
        return




# test
# #ax = plt.subplot(111)
# files = os.listdir(os.path.curdir)
# for f in files:
#     if f.endswith('.json'):
#         ax = plt.subplot(111)
#         newfile = esr_data(f)
#         newfile.plot_data(ax)
#         plt.show()


