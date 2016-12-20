__author__ = 'VADIM'

__author__ = 'Vadim'

import json
from matplotlib import pyplot as plt
import numpy as np
import os
from pylab import *
from scipy import optimize

def RABI_OSCILATIONS(level,intensity, freq, T2_star,fi):

    return np.vectorize(lambda t: level*(1.0 - intensity*0.5*(np.cos(2*np.pi*freq*t+fi))*np.exp(-t/T2_star)))

def fitrabi(data):
    params = [1e-1,0.1, 0.001, 2000.0,0.0]
    errorf = lambda p: ravel(RABI_OSCILATIONS(*p)(data[0])-data[1])
    p,success = optimize.leastsq(errorf, params, maxfev = 1000)
    return p

class rabi_data():
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
        ts = []
        counts = []
        try:
            for row in self.data[:]:
                t = row[4]
                if t<50:
                    continue
                ts.append(t)
                counts.append(row[-1]/row[-2])
        except:
            print('Oops, data file does not exist')


        #plt.show()
        if fit:
            self.params = fitrabi([ts,counts])
            fitF = RABI_OSCILATIONS(*self.params)
            print(self.params)
            fitplotfreq = np.arange(ts[0],ts[-1],(ts[-1]-ts[0])/1000.0)
            #print(fitplotfreq)
            fitdata = fitF(fitplotfreq)
            #fitdata = np.zeros(fitplotfreq.shape)
            print('fit!')
            label = 'Rabi data fit\n F={0:.2f} MHz\nT2*={1:.0f} ns\nMW = {2:.0f}dBm'.format(self.params[2]*1e3,self.params[3],self.data[0][1])
            ax.plot(fitplotfreq,fitdata,'k-',lw=4, label = label)
            ax.plot(ts, counts,formats[plot_id%len(formats)], label = self.filetail[0:-5])
        else:
            ax.plot(ts, counts,formats[plot_id%len(formats)], label = self.filetail[0:-5])
        if legend:
            plt.legend(loc=loc)

    def file_parsing(self):
        data_file = open(self.filename,'r')
        data = json.load(data_file)
        if data["modulation_scheme"]!="Rabi oscillation":
            raise Exception("Wrong scheme")
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


