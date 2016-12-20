__author__ = 'Alexey'
from lmfit import  minimize, Parameters
from numpy import sin,exp
params = Parameters()

def residual(params, x, data, eps_data):
    amp = params['amp'].value
    pshift = params['phase'].value
    freq = params['frequency'].value
    decay = params['decay'].value
    model = amp * sin(x * freq  + pshift) * exp(-x*x*decay)

    return (data-model)/eps_data

def func_in_func(x, ):

    def printer(freq,x):
        print(tata)
    printer(x)
    return

func_in_func(10)