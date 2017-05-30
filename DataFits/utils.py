import numpy
from scipy.fftpack import fft
import numpy as np

def get_power_spectrum(xs,ys,maxF = 5.,df= 0.05,exclude_DC=True):
    ts = xs
    dt = ts[1]-ts[0]
    n_fourier = max(int(1./df/dt),len(ys))
    n_fourier += n_fourier%1+1 #odd data count
    if exclude_DC:
        ysw = fft(ys - np.mean(ys),n_fourier)
    else:
        ysw = fft(ys,n_fourier)
    power_spectrum = ysw[1:]
    power_spectrum = abs(power_spectrum)**2
    power_spectrum += power_spectrum[::-1]
    fs = np.arange(1,len(ysw))/dt/n_fourier
    power_spectrum = power_spectrum[np.nonzero(fs<maxF)]
    fs = fs[np.nonzero(fs<maxF)]
    return fs,power_spectrum