from .DataFit import DataFit
from scipy import optimize
from pylab import ravel
__author__ = 'Vladimir'

import numpy as np
from numpy import exp,cos
import scipy as scp
from matplotlib import pyplot as plt
from scipy import signal as s
from scipy.fftpack import fft,fftfreq,fftshift
#for single NV

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
    fs = np.arange(len(ysw))/dt/n_fourier
    power_spectrum = power_spectrum[np.nonzero(fs<maxF)]
    fs = fs[np.nonzero(fs<maxF)]
    return fs,power_spectrum

class RabiFit1(DataFit): #single
    plot_power_spectrum = True
    plot_envelope = True
    x_row = 4 # T pulse
    counts_type=1
    fit_params_description = ['level',
                              'Contrast',
                              'Rabi frequency',
                              'T2_star',
                              'fi']
    label_fit_params = [1,2,3]
    #label_data_params = [0,1] # frequency, power #added in label_additional info
    def initial_guess(self,x_data,y_data): #Must be defined by user
        level = max(y_data)
        fs,pwrs = get_power_spectrum(x_data*1e-3,y_data,df=0.1,maxF=2.)
        f = fs[np.argmax(pwrs[1:])+1] #exclude 0 freq
        #print('Rabi fit: guessed F={0}'.format(f))
        return [level,1, f, 1000.0,0.0]
    #def parameter_scoring(self,*params):
     #   freq0=0.5
      #  freq1=8.5
       # f = params[2]
       # if freq0<f<freq1:
       #     return 0.
       # else:
       #     return 1.
    def label_additional_info(self):
        freq = self.fit_parameters[2]*1e-3 #in Ghz/ns
        fi = self.fit_parameters[4]
        pi_length = (0.5 - fi/2./np.pi)/freq
        pi_length_uncomp = (0.5)/freq
        pi_comp = (-fi/2./np.pi)/freq
        return ['MW: {0:.2f} @ {1:.1f}dBm'.format(*self.first_row[:2]),
                'Pi= {0:.1f} ns'.format(pi_length),
                'Echo: pi= {0:.1f} ns, pi_comp={1:.1f} ns'.format(pi_length_uncomp, pi_comp)]
    def fit_fun(self,t,level,intensity, freq, T2_star,fi): #Must be defined by user
        return level*(1.0 - intensity*0.5*(np.cos(2*np.pi*freq*1e-3*t+fi))*np.exp(-t/T2_star))
        # tau = t/T2_star
        # return level*exp(-tau)*(0.05+0.8*exp(tau)+0.15*cos(2*np.pi*t*freq*1e-3+fi))
    def envelopes(self,t0,t1,n_points=200):
        ts = np.linspace(t0,t1,n_points)
        level = self.fit_parameters[0]
        intensity = self.fit_parameters[1]
        T2_star= self.fit_parameters[3]
        env_p = level*(1.0 - intensity*0.5*np.exp(-ts/T2_star))
        env_m = level*(1.0 + intensity*0.5*np.exp(-ts/T2_star))
        return ts,env_p,env_m
    def fitted_data(self, n_points=1000):
        fit_f = self.build_fit(*(self.fit_parameters))

        #self.y_data /= self.fit_parameters[0] #Some black magic. Normalization??
        #self.y_data += 0.5*self.fit_parameters[1]

        x_min = np.min(self.x_data)
        x_max = np.max(self.x_data)
        fit_x = np.linspace(x_min,x_max,n_points)
        fit_y = fit_f(fit_x)
        #fit_y = (fit_f(fit_x) / self.fit_parameters[0])+0.5*self.fit_parameters[1] if len(fit_x) > 0 else []
        return fit_x, fit_y
    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        c_data = '#7f7f7f'
        c_fit = '#ff7f0e'


        #data_format = c + '+-' if data_format is None else data_format
        #fit_format = c + '-' if fit_format is None else fit_format
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(self.x_data, self.y_data, '-', color=c_data, lw=1)
        axes.plot(self.x_data, self.y_data, 'o', color=c_data, ms=2)
        axes.plot(x_data_fit, y_data_fit, '-', color = c_fit, lw=3, label=self.compose_label())
        #axes.plot(self.x_data, self.y_data, data_format)
        axes.set_xlabel('MW duration, ns')
        axes.set_ylabel(self.plot_y_label())
        leg = axes.legend(loc='upper right')
        leg.get_frame().set_alpha(0.5)
        t0 = min(0,-self.fit_parameters[4]/(2.*np.pi*self.fit_parameters[2]*1.e-3))
        axes.set_xlim((t0,max(x_data_fit)))
        print(t0)
        if self.plot_power_spectrum:
            self.plot_fft(axes,self.x_data,self.y_data)
        if self.plot_envelope:
            ts,env_p,env_m = self.envelopes(t0,max(x_data_fit))
            env_params = dict(color='k',alpha=0.2,lw=2.5)
            axes.plot(ts,env_p,'-',**env_params)
            axes.plot(ts,env_m,'-',**env_params)

        if min(x_data_fit)>0.0:
            fit_f = self.build_fit(*(self.fit_parameters))
            ts = np.linspace(t0,min(x_data_fit),100)
            axes.plot(ts,fit_f(ts),'--', color = c_fit, lw=3)
    def plot_fft(self,axes,xs,ys):
        fs,power_spectrum = get_power_spectrum(xs*1e-3,ys,df=0.1,maxF=5.)
        sp_plot = plt.axes([0.6,0.2,0.3,0.1])
        sp_plot.set_xlabel('F, Mhz')
        sp_plot.set_ylabel('Power')
        sp_plot.set_alpha(0.5)
        sp_plot.set_yscale('log')
        sp_plot.get_yaxis().set_ticks([])
        sp_plot.plot(fs,power_spectrum)
#for the ensemble
class RabiFit(DataFit): # ens
    x_row = 4 # T pulse
    counts_type=1
    fit_params_description = ['level',
                              'intensity',
                              'w Rabi, Mhz',
                              'T2*,ns',
                              'T1,ns']
    label_fit_params = [1,2,3,4]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.99,0.008, 1, 1000.0,1000000.0]
    def fit_fun(self,t,level,intensity, freq, T2_star,T3_star): #Must be defined by user
        return level*((intensity*(0.5*(1+cos(2*np.pi*freq*1e-3*t-30)))*exp(-t/T2_star)+1-intensity)*exp(-t/T3_star)+0.7)

class PiPulseFit(DataFit):
    x_row = 4 #Pi time
    counts_type = 0 # contrast type
    fit_params_description = ['noise level',
                              'Contrast',
                              'Pi pulse, ns',
                              'Front compensation',]
    label_fit_params = [2,3,1]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [np.mean(y_data),np.max(y_data)-np.min(y_data), 70.0,0.0,700]
    def fit_fun(self,t,level,contrast, pi_tau, comp, t_coh): #Must be defined by user
        return level + contrast*np.exp(-t/t_coh)*np.power(np.cos(np.pi*(t+comp)/pi_tau/2),2)

class ExcitationCollectionAlignmentFit(DataFit):
    x_row = 3 #Shift
    counts_type=0
    fit_params_description = ['noise level',
                              'half_width',
                              'height',
                              't0']
    label_fit_params = [3]
    label_data_params = [] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [np.mean(y_data), 2000.0, np.max(y_data),np.mean(x_data)]
    def fit_fun(self,t,noise_level, half_width, height, t0): #Must be defined by user
        d = abs(t-t0)
        if d>half_width :
            return noise_level
        return noise_level + (1-abs(t-t0)/half_width)*height

class ESRFit2(DataFit):
    x_row = 0 # Frequency
    counts_type=1
    label_fit_params = [0,1,2,3]
    fit_params_description = ['Width',
                              'Alpha',
                              'Frequency',
                              ]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [10, 0.5 ,np.mean(x_data)]
    def fit_fun(self,freq,width,alpha, position): #Must be defined by user
        return 1 - (width*alpha)/((freq-position)**2+(width**2)/4)

class ESRFitMono(DataFit):
    x_row = 0 # Frequency
    counts_type=1
    label_fit_params = [0,3,2,5]
    fit_params_description = ['Width',
                              'Alpha',
                              'Frequency',
                              'Width2',
                              'Alpha2',
                              'Frequency2'
                              ]
    label_data_params = [1] # frequency, power

    data_format = 'k-+'
    fit_format = 'r-'

    def initial_guess(self,x_data,y_data): #Must be defined by user

        f = []
        f_step = x_data[1]-x_data[0]
        f_window = 5 # Mhz
        number =int(f_window/f_step)
        for i in range(len(x_data)-number):
            f.append(sum(y_data[i:i+number]))
        try:
            i1 = number/2 + np.argmin(np.array(f))
        except:
            i1 = int(len(x_data)/2)
        return [5, 0.5 ,x_data[i1],5, 0.2 ,x_data[i1]+5]

    def fit_fun(self,freq,w1,a1,p1,w2,a2,p2): #Must be defined by user
        lor = 0
        lor = (w1*a1)/((freq-p1)**2+(w1**2)/4)
        lor = lor + (w2*a2)/((freq-p2)**2+(w2**2)/4)
        return 1 - lor

# For dynamic number of peaks try this \/

class ESRFit(DataFit):

    x_row = 0 # Frequency
    counts_type=1
    label_fit_params = []
    fit_params_description = []
    label_data_params = [1] # frequency, power

    data_format = 'k-+'
    fit_format = 'r-'

    def initial_guess(self,x_data,y_data): #Must be defined by user

        f = []
        f_step = x_data[1]-x_data[0]
        debug = False
        # adding a tail for data from the end
        num = 50
        y_data = np.append(y_data,np.random.normal(1,0.004,num))
        x_data = np.append(x_data,np.arange(x_data[-1],x_data[-1]+f_step*num,f_step))
        #widthtypical = int(20/f_step)
        f_window = 5# Mhz
        number =int(f_window/(2*f_step))
        for i in range(len(x_data)-number):
            f.append(sum(y_data[i-number:i+number])/(2*number))
        ff = np.array(f)
        #xs = s.find_peaks_cwt(np.array(-y_data),widths=np.arange(1,15),min_snr=5,noise_perc=30)
        xs = s.find_peaks_cwt(np.array(-y_data),widths=np.arange(1,15),min_snr=2.5,noise_perc=20)
        #plt.plot(np.diff(ff)[number:],'ko-')
        if debug:
            plt.plot(x_data,y_data,'go-')
            plt.plot(x_data[xs],y_data[xs],'ro')
            plt.show()
        #print(xs,'from findpeaks')

        widths = [3 for x in range(len(xs))]
        powers = [0.2 for x in range(len(xs))]
        pos = [x_data[i] for i in xs]

        return widths + powers + pos





    def fit_fun(self,freq,*params): #Must be defined by user
        lor = 0
        n = int(len(params)/3)
        widths = params[0:n]
        alphas = params[n:2*n]
        positions = params[2*n:3*n]
        for i in range(0,n):
            lor +=  (widths[i]*alphas[i])/((freq-positions[i])**2+(widths[i]**2)/4)
        return 1 - lor

    def try_fit(self, x_data, y_data):
        error = 0
        params = np.array(self.initial_guess(x_data, y_data))

        if params == []:
            self.fit_parameters = np.array([])
            self.fit_parameters_error = np.array([])
        else:
            errorf = lambda p: ravel(self.build_fit(*p)(x_data) - y_data)
            # p, success = optimize.leastsq(errorf, params, maxfev=1000)
            # self.fit_parameters = p
            #             self.fit_error = errorf(p) / float(len(x_data))
            pfit, pcov, infodict, errmsg, success = optimize.leastsq(errorf, params, maxfev=1000,full_output=True)
            self.fit_parameters = pfit
            self.fit_error = errorf(pfit) / float(len(x_data))
            if pcov is not None:
                sum_squares = (errorf(pfit)**2).sum() / float(len(x_data)-len(pfit))
                self.fit_parameters_error = np.diag(abs(pcov*sum_squares))**0.5
            else:
                self.fit_parameters_error = np.zeros(self.fit_parameters.shape)
        # self.fit_success = success!=
        #return success
        self.n = int(len(pfit)/3)
        self.widths = pfit[0:self.n]
        self.alphas = pfit[self.n:2*self.n]
        self.positions = pfit[2*self.n:3*self.n]

    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        #c = rand_color()
        data_format = self.data_format if data_format is None else data_format
        fit_format = self.fit_format if fit_format is None else fit_format
        data_format = 'k+-' + '+-' if data_format is None else data_format
        fit_format = 'r-' + '-' if fit_format is None else fit_format
        axes.plot(self.x_data, self.y_data, data_format)
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(x_data_fit, y_data_fit, fit_format, lw=3, label=self.compose_label())
        axes.plot(self.x_data, self.y_data, data_format)

        axes.set_xlabel(self.plot_x_label())
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='center')
        if max(self.x_data)>2870.0 and min(self.x_data)<2870.0:
            axes.plot([2870,2870],[0,2],'r--')
        for i in range(self.n):
            axes.text(self.positions[i]+0.5,
                      -0.002+self.fit_fun(self.positions[i],
                                                           *self.fit_parameters),'{0:.1f}\n({3:.1f})'
                                                                                 '({1:.2f})\n{2:.1f}G'.format(self.positions[i],self.widths[i],abs(self.positions[i]-2870.0)/2.8,self.fit_parameters_error[2*self.n+i]),color='r')

        axes.text(np.mean(self.x_data),max(self.y_data),comment)
        axes.set_ylim(min(self.y_data)*0.9995,max(self.y_data)*1.0001)


class EchoFit4(DataFit):
    x_row = 5 # T pulse
    counts_type = 2
    fit_params_description = ['intensity',
                              'T2',
                              'level',
                              'N']
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'


    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.1,1000,0.0,2]
    def fit_fun(self,t,intensity, T2,level,N): #Must be defined by user
        return level+(intensity*np.exp(-(t/T2)**N))
    def x_fun(self, row):
        #4 -> Pi time 6-> Pi compensation
        pi_half = row[4]/2. + row[6]
        pi = row[4] + row[6]
        echo_correction = pi_half*2 + pi #pi_time
        return row[self.x_row]*2+echo_correction
    def y_fun(self, row):
        return (row[-1] - row[-2]) / (row[-1] + row[-2])*2.0 if (row[-1] + row[-2]) != 0.0 else -1.0

    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        c = 'r'
        data_format = self.data_format if data_format is None else data_format
        fit_format = self.fit_format if fit_format is None else fit_format
        data_format = c + '+-' if data_format is None else data_format
        fit_format = c + '-' if fit_format is None else fit_format
        axes.plot(self.x_data, self.y_data, data_format)
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(x_data_fit, y_data_fit, fit_format, lw=3, label=self.compose_label())
        #axes.plot(self.x_data, self.y_data, data_format)
        signal = self.all_data[:,-1]
        reference = self.all_data[:,-2]
        norm = np.mean((signal+reference)/2.)

        signal/=norm
        reference/=norm
        #axes.plot(self.x_data,signal,lw=2,alpha=0.5,color='g',label='Signal')
        #axes.plot(self.x_data,reference,lw=2,alpha=0.5,color='r',label='Reference')
        axes.set_xlabel(self.plot_x_label())
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='best')

class EchoFit2(DataFit):
    x_row = 5 # T pulse
    counts_type = 1
    fit_params_description = ['intensity',
                              'T2_star',
                              'T2_thresh']
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'


    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.1,2000,3000]
    def fit_fun(self,t,intensity, T2_star,T2_thresh): #Must be defined by user
        return 1+(intensity/(1+np.exp((t-T2_thresh)/T2_star)))

class EchoFit3(DataFit):
    x_row = 5 # T pulse
    counts_type = 1
    fit_params_description = ['intensity',
                              'T2_star',
                              'T2_thresh']
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'


    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.1,500,1.0]
    def fit_fun(self,t,intensity, T2_star,T2_thresh): #Must be defined by user
        return T2_thresh+(intensity*np.exp(-(t/T2_star)**2))

class EchoFit(DataFit):
    x_row = 5 # T pulse
    counts_type = 1
    fit_params_description = ['intensity',
                              'T2',
                              'level',
                              'N']
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'


    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.1,1000,1.0,2]
    def fit_fun(self,t,intensity, T2,level,N): #Must be defined by user
        return level+(intensity*np.exp(-(t/T2)**N))
    def x_fun(self, row):
        return row[self.x_row]


class DynamicScheme(DataFit):
    x_row = 2 # duration iter
    counts_type=1
    fit_params_description = ['Level',
                              'Contrast',
                              'T1, time steps']
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [1,0.07, 1500.0]
    def fit_fun(self,t,level,intensity,T1): #Must be defined by user
        return level*(1+intensity*exp(-t/T1))

    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        c_data = '#7f7f7f'
        c_fit = '#ff7f0e'


        #data_format = c + '+-' if data_format is None else data_format
        #fit_format = c + '-' if fit_format is None else fit_format
        x_data_fit, y_data_fit = self.fitted_data()
        print(max(self.y_data))
        axes.plot(self.x_data, self.y_data, '-', color=c_data, lw=1)
        axes.plot(self.x_data, self.y_data, 'o', color=c_data, ms=2)
        axes.plot(x_data_fit, y_data_fit, '-', color = c_fit, lw=3, label=self.compose_label())
        #axes.plot(self.x_data, self.y_data, data_format)
        axes.set_xlabel('Idle duration, timesteps')
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='center')


class T1Fit(DataFit):
    x_row = 5 # T pulse
    counts_type = 2
    fit_params_description = ['intensity',
                              'T2',
                              'level',
                              'N']
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'


    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.1,1000,0.0,2]
    def fit_fun(self,t,intensity, T2,level,N): #Must be defined by user
        return level+(intensity*np.exp(-(t/T2)**N))
    def x_fun(self, row):
        #4 -> Pi time 6-> Pi compensation
        pi_half = row[4]/2. + row[6]
        pi = row[4] + row[6]
        echo_correction = pi_half*2 + pi #pi_time
        return row[self.x_row]*2+echo_correction
    def y_fun(self, row):
        return (row[-1] - row[-2]) / (row[-1] + row[-2])*2.0 if (row[-1] + row[-2]) != 0.0 else -1.0

    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        c = 'r'
        data_format = self.data_format if data_format is None else data_format
        fit_format = self.fit_format if fit_format is None else fit_format
        data_format = c + '+-' if data_format is None else data_format
        fit_format = c + '-' if fit_format is None else fit_format
        axes.plot(self.x_data, self.y_data, data_format)
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(x_data_fit, y_data_fit, fit_format, lw=3, label=self.compose_label())
        #axes.plot(self.x_data, self.y_data, data_format)
        signal = self.all_data[:,-1]
        reference = self.all_data[:,-2]
        norm = np.mean((signal+reference)/2.)

        signal/=norm
        reference/=norm
        #axes.plot(self.x_data,signal,lw=2,alpha=0.5,color='g',label='Signal')
        #axes.plot(self.x_data,reference,lw=2,alpha=0.5,color='r',label='Reference')
        axes.set_xlabel(self.plot_x_label())
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='best')
