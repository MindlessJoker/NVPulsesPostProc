from pylab import ravel
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
#from DataFits.Rabi import RabiFit1
from DataFits.utils import get_power_spectrum
from .DataFit import DataFit
from itertools import product
__author__ = 'Vladimir'

from numpy import exp,cos
import matplotlib
import matplotlib.axes
from scipy import optimize as opt
import matplotlib.gridspec as gridspec
from scipy import signal as s
import scipy
#from .Rabi import *
#for single NV




#for the ensemble

class RamseyFit(DataFit):
    x_row = 7
    counts_type = 1

    fit_params_description = ['level',
                              'a1',
                              'a2',
                              'a3',
                              'central detuning, Mhz',
                              'T2*,ns',
                              'phi']
    label_fit_params = [0,1,2,3,4,5,6]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [1,0.1,0.1,0.1, 0.05, 500.0,0.0]
    def fit_fun(self,t,level,a1,a2,a3, freq, T2_star,phi): #Must be defined by user

        freq0 = freq
        freq1 = 2.200-freq
        freq2 = freq0 + 2.200

        sumofcos = abs(a1)*cos(2*np.pi*freq0*1e-3*t-phi)+\
                   abs(a2)*cos(2*np.pi*freq1*1e-3*t-phi)+\
                   abs(a3)*cos(2*np.pi*freq2*1e-3*t-phi)

        return (abs(level)-sumofcos*exp(-t/abs(T2_star)))

    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        #c = rand_color()
        data_format = self.data_format if data_format is None else data_format
        fit_format = self.fit_format if fit_format is None else fit_format
        data_format = 'k+-' + '+-' if data_format is None else data_format
        fit_format = 'b-' + '-' if fit_format is None else fit_format
        axes.plot(self.x_data, self.y_data, 'k+-')
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(x_data_fit, y_data_fit, fit_format, lw=3, label=self.compose_label())
        axes.plot(self.x_data, self.y_data, 'r-')
        axes.set_xlabel(self.plot_x_label())
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='best',framealpha=0.4)
        # if max(self.x_data)>2870.0 and min(self.x_data)<2870.0:
        #     axes.plot([2870,2870],[0,2],'r--')
        # for i in range(self.n):
        #     axes.text(self.positions[i]+0.5,
        #               -0.002+self.fit_fun(self.positions[i],
        #                                                    *self.fit_parameters),'{0:.1f}\n({3:.1f})'
        #                                                                          '({1:.2f})\n{2:.1f}G'.format(self.positions[i],self.widths[i],abs(self.positions[i]-2870.0)/2.8,self.fit_parameters_error[2*self.n+i]),color='r')

        # axes.text(np.mean(self.x_data),max(self.y_data),comment)
        # axes.set_ylim(min(self.y_data)*0.9995,max(self.y_data)*1.0001)



class RabiFit(DataFit): # ens
    x_row = 'T pulse'
    counts_type=1
    fit_params_description = ['level',
                              'intensity',
                              'w Rabi, Mhz',
                              'T2*,ns',
                              'T1,ns']
    label_fit_params = [1,2,3,4]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.99,0.008, 6, 1000.0,1000000.0]
    def fit_fun(self,t,level,intensity, freq, T2_star,T3_star): #Must be defined by user
        return level*((intensity*(0.5*(1+cos(2*np.pi*freq*1e-3*t-30)))*exp(-t/T2_star)+1-intensity)*exp(-t/T3_star)+0.7)

class RabiFit1(DataFit): #single
    max_frequency = 20.2
    data_mean_window = 1
    fit_max_freq = 20.
    x_row = 'T pulse'
    counts_type=1
    fit_params_description = ['level',
                              'Contrast',
                              'Rabi frequency',
                              'T2_star',
                              'fi']
    label_fit_params = [1,2,3,4]
    #label_data_params = [0,1] # frequency, power #added in label_additional info
    def __init__(self,*args,**kwargs):

        if self.data_mean_window>1:
            wl = (self.data_mean_window//2)*2+1
            w = np.ones(wl)/wl
            args = list(args)
            args[0] = np.array(args[0])
            args[0][:,-1] = np.convolve(args[0][:,-1],w,'same')
            args[0][:,-2] = np.convolve(args[0][:,-2],w,'same')
        super(RabiFit1,self).__init__(*args,**kwargs)
        self.plot_power_spectrum = True
        self.plot_envelope = True
        if self.data_mean_window>1 and False:
            wl = (self.data_mean_window//2)*2+1
            w = np.ones(wl)/wl
            self.y_data = np.convolve(self.y_data,w,'valid')
            cut = self.data_mean_window//2
            self.x_data = self.x_data[cut:-cut]
    def initial_guess(self,x_data,y_data): #Must be defined by user
        rabi_tresh = self.fit_max_freq
        level = max(y_data)
        fs,pwrs = get_power_spectrum(x_data*1e-3,y_data,df=0.1,maxF=self.max_frequency)
        indexes = np.arange(fs.shape[0])[fs < rabi_tresh]

        f = fs[np.argmax(pwrs[1:indexes[-1]])+1] #exclude 0 freq
        print(f)
        #print('Rabi fit: guessed F={0}'.format(f))
        return [level,0.1, f, 1000.0,0.0]
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
        if fi<0:
            fi+=2.*np.pi
        pi_length = (0.5 - fi/2./np.pi)/freq
        pi_length_uncomp = (0.5)/freq
        pi_comp = (-fi/2./np.pi)/freq
        return ['MW: {0:.2f} @ {1:.1f}dBm'.format(*self.first_row[:2]),
                'Pi= {0:.1f} ns'.format(pi_length),
                'Echo: pi= {0:.1f} ns, pi_comp={1:.1f} ns'.format(pi_length_uncomp, pi_comp)]
    def fit_fun(self,t,level,intensity, freq, T2_star,fi): #Must be defined by user
        return level*(1.0 - abs(intensity)*0.5*(np.cos(2*np.pi*abs(freq)*1e-3*t+fi+np.pi))*np.exp(-t/T2_star))
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

        axes.plot(x_data_fit, y_data_fit, '-', color = c_fit, lw=3, label=self.compose_label(),alpha=0.4)
        #axes.plot(self.x_data, self.y_data, data_format)
        axes.set_xlabel('MW duration, ns')
        axes.set_ylabel(self.plot_y_label())
        leg = axes.legend(loc='upper right',framealpha=0.4)
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
            axes.plot(ts,fit_f(ts),'--', color = c_fit, lw=3,alpha=0.4)
        axes.plot(self.x_data, self.y_data, '-', color=c_data, lw=1)
        axes.plot(self.x_data, self.y_data, 'o', color=c_data, ms=2)
    def plot_fft(self,axes,xs,ys):
        fs,power_spectrum = get_power_spectrum(xs*1e-3,ys,df=0.1,maxF=13.)
        sp_plot = plt.axes([0.6,0.2,0.3,0.1])
        sp_plot.set_xlabel('F, Mhz')
        sp_plot.set_ylabel('Power')
        sp_plot.set_alpha(0.5)
        sp_plot.set_yscale('log')
        sp_plot.get_yaxis().set_ticks([])
        sp_plot.plot(fs,power_spectrum)

class RabiFit3sin(RabiFit1): #single
    plot_power_spectrum = True
    plot_envelope = True


    fit_params_description = ['level',
                              'Contrast',
                              'Rabi frequency',
                              'T2_star',
                              'fi']
    label_fit_params = [1,2,3]
    #label_data_params = [0,1] # frequency, power #added in label_additional info
    def initial_guess(self,x_data,y_data): #Must be defined by user

        params = super(RabiFit3sin,self).initial_guess(x_data,y_data)
        print(np.sqrt(params[2]**2-2.2**2))
        return [params[0],params[1], params[2],
                params[3],params[4],
                np.sqrt(params[2]),
                np.sqrt(params[2]**2-2.2**2),
                0.3, 0.3, 0.3]

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
    def fit_fun(self,t,level,intensity, freq, T2_star,fi,freq2, freq3, A1, A2, A3): #Must be defined by user
        return level*(1.0 - intensity*0.5*(
            A1*np.cos(2*np.pi*freq*1e-3*t+fi)+
            A2*np.cos(2*np.pi*freq2*1e-3*t+fi)+
            A3*np.cos(2*np.pi*freq3*1e-3*t+fi))*np.exp(-t/T2_star))
        # tau = t/T2_star
        # return level*exp(-tau)*(0.05+0.8*exp(tau)+0.15*cos(2*np.pi*t*freq*1e-3+fi))



class RabiFit3sin2(RabiFit1): #single
    plot_power_spectrum = True
    plot_envelope = True


    fit_params_description = ['level',
                              'Contrast',
                              'Rabi frequency',
                              'T2_star',
                              'fi',
                              'Center det',
                              'A-1',
                              'Amp+1']
    label_fit_params = [1,2,3]
    #label_data_params = [0,1] # frequency, power #added in label_additional info
    def initial_guess(self,x_data,y_data): #Must be defined by user

        params = super(RabiFit3sin2,self).initial_guess(x_data,y_data)
        return [params[0],params[1], params[2],
                params[3],0.,
                0., 0., 0.]

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
    def fit_fun(self,t,level,intensity, freq, T2_star,fi,center_det, A1, A2): #Must be defined by user
        res_dist = 2.2
        Am1 = np.min([0.5,np.abs(A1)])
        Am2 = np.min([0.5,np.abs(A2)])
        freq0 = np.sqrt(freq**2+center_det**2)
        freq_m1 = np.sqrt(freq**2+(abs(center_det)-res_dist)**2)
        freq_p1 = np.sqrt(freq**2+(abs(center_det)+res_dist)**2)
        A0 = 1.-A1-A2
        return level*(1.0 + intensity*0.5*(
            A0 *np.cos(2*np.pi*freq0*1e-3*(t+fi))+
            Am1*np.cos(2*np.pi*freq_m1*1e-3*(t+fi))+
            Am2*np.cos(2*np.pi*freq_p1*1e-3*(t+fi)))*np.exp(-t/abs(T2_star)))

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
    x_row = 3 #Shift ????
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
    x_row = 'MW Frequency' #0
    counts_type=1
    label_fit_params = [0,1,2,3]
    fit_params_description = ['Width',
                              'Alpha',
                              'Frequency',
                              ]
    label_data_params = [0,1] # frequency, power
    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [7, 0.5 ,np.mean(x_data)]
    def fit_fun(self,freq,width,alpha, position): #Must be defined by user
        return 1 - (width*alpha)/((freq-position)**2+(width**2)/4)

class ESRFitMono(DataFit):
    x_row = 'MW Frequency' #0
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
        f_window = 4 # Mhz
        number =int(f_window/f_step)
        for i in range(len(x_data)-number):
            f.append(sum(y_data[i:i+number]))
        try:
            i1 = number/2 + np.argmin(np.array(f))
        except:
            i1 = int(len(x_data)/2)
        return [7, 0.5 ,x_data[i1],5, 0.2 ,x_data[i1]+5]

    def fit_fun(self,freq,w1,a1,p1,w2,a2,p2): #Must be defined by user
        lor = 0
        lor = (w1*a1)/((freq-p1)**2+(w1**2)/4)
        lor = lor + (w2*a2)/((freq-p2)**2+(w2**2)/4)
        return 1 - lor

# For dynamic number of peaks try this \/

class ESRFit(DataFit):

    x_row = 'MW Frequency' # 0
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
        axes.legend(loc='center',framealpha=0.4)
        if max(self.x_data)>2870.0 and min(self.x_data)<2870.0:
            axes.plot([2870,2870],[0,2],'r--')
        for i in range(self.n):
            axes.text(self.positions[i]+0.5,
                      -0.002+self.fit_fun(self.positions[i],
                                                           *self.fit_parameters),'{0:.1f}\n+-({3:.1f})\n'
                                                                                 'width({1:.2f})\n{2:.1f}G'.format(self.positions[i],self.widths[i],abs(self.positions[i]-2870.0)/2.8,self.fit_parameters_error[2*self.n+i]),color='r')

        axes.text(np.mean(self.x_data),max(self.y_data),comment)
        axes.set_ylim(min(self.y_data)*0.9995,max(self.y_data)*1.0001)

class ESRFit_pulsed(DataFit):

    x_row = 'MW Frequency' # 0
    counts_type=1
    label_fit_params = [0,1,2,3]
    fit_params_description = ['f0','FWHM','contrast','scale']
    label_data_params = [1] # frequency, power

    data_format = 'k-+'
    fit_format = 'r-'


    def fitfun0(self,f,f0,fwhm,contrast,scale):
        return scale*(1-(abs(contrast)*(fwhm/2)**2/((f-f0)**2+(fwhm/2)**2)))

    def fit_fun(self,f,*params):#f0,fwhm,contrast,scale,p1,p2):

        return params[3]*(1-(abs(params[2])*abs(params[4])*(params[1]/2)**2/((f-params[0])**2+(params[1]/2)**2))-
                       (abs(params[2])*abs(params[4])*(params[1]/2)**2/((f-params[0]-2.2)**2+(params[1]/2)**2))-
                       (abs(params[2])*abs((1-abs(params[4])-abs(params[5])))*(params[1]/2)**2/((f-params[0]+2.20)**2+(params[1]/2)**2)))


    def initial_guess(self, x_data, y_data):
        f0 = self.x_data[list(self.y_data).index(min(self.y_data))]
        scale = self.y_data[-1]
        contrast = 0.01
        width = 0.2

        params = [f0,width,contrast,scale]
        popt,pcov = opt.curve_fit(self.fitfun0,self.x_data,self.y_data,p0=params)
        f0 = popt[0]
        print(f0)

        return [f0,1,contrast,scale,0.33,0.33]


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
        print(self.fit_parameters)
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
        #axes.legend(loc='center',)
        axes.legend(loc=2,framealpha=0.4)
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
    x_row = 'T echo'
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
        pi_time = self.get_data_from_row(row,'PI time')
        pi_comp = self.get_data_from_row(row,'PI compensation')
        x_data = self.get_data_from_row(row,self.x_row)
        #4 -> Pi time 6-> Pi compensation
        pi_half = pi_time/2. + pi_comp
        pi = pi_time + pi_comp
        echo_correction = pi_half*2. + pi #pi_time
        return x_data*2.+echo_correction
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
        axes.legend(loc='best',framealpha=0.4)

class EchoFit2(DataFit):
    x_row =  'T pulse'
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
    x_row = 'T pulse' #5+2
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
    x_row = 'T pulse' # 5 + 2
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
class EchoFitRevival(EchoFit4):
    x_row = 'T echo' # 5 + 2
    counts_type = 1
    fit_params_description = ['intensity1',
                              'N2',
                              'T2',
                              'level',
                              'N',
                              'Rev_freq']
    label_fit_params = [0,1,2,3,4,5]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'


    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.02,2.,50000,.0,3.,50]
    def fit_fun(self,t,intensity1, N2, T2,level,N, freq): #Must be defined by user#
        return level+((0.5+0.5*np.cos(2*np.pi*abs(freq)*t/1e6))**np.min([abs(N2),3.]))*intensity1*np.exp(-(t/T2)**N)
    def envelope(self,t,intensity1, N2, T2,level,N, freq):
        return level+(1.*intensity1*np.exp(-(t/T2)**N))

    def envelopes(self,t0,t1,n_points=200):
        ts = np.linspace(t0,t1,n_points)
        intensity = self.fit_parameters[0]
        #intensity = self.fit_parameters[1]
        T2 = self.fit_parameters[2]
        level = self.fit_parameters[3]
        N = self.fit_parameters[4]
        env_p = level+((0.5+0.5*1)*intensity*np.exp(-(ts/T2)**N))
        env_m = level+((0.5-0.5*1)*intensity*np.exp(-(ts/T2)**N))
        return ts,env_p,env_m
    def plot_fft(self,axes,xs,ys):
        fs,power_spectrum = get_power_spectrum(xs*1e-3,ys,df=0.1,maxF=13.)
        sp_plot = plt.axes([0.6,0.2,0.3,0.1])
        sp_plot.set_xlabel('F, Mhz')
        sp_plot.set_ylabel('Power')
        sp_plot.set_alpha(0.5)
        sp_plot.set_yscale('linear')
        sp_plot.get_yaxis().set_ticks([])
        sp_plot.plot(fs,power_spectrum)
    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):

        import json
        #super(EchoFitRevival,self).plot_me(axes,data_format,fit_format,comment)
        data = {}

        c = 'r'
        data_format = self.data_format if data_format is None else data_format
        fit_format = self.fit_format if fit_format is None else fit_format
        data_format = c + '+-' if data_format is None else data_format
        fit_format = c + '--' if fit_format is None else fit_format
        axes.plot(self.x_data, self.y_data, data_format)

        data['data'] = [self.x_data.tolist(), self.y_data.tolist()]
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(x_data_fit, y_data_fit, fit_format, lw=1, label=self.compose_label())

        data['fit'] = [list(x_data_fit), list(y_data_fit)]

        t0  = min(0,-self.fit_parameters[4]/(2.*np.pi*self.fit_parameters[2]*1.e-3))
        ts,env_p,env_m = self.envelopes(t0,max(x_data_fit))
        env_params = dict(color='r',alpha=0.8,lw=2.5)
        axes.plot(ts,env_p,'-',**env_params)
        axes.plot(ts,env_m,'-',**env_params)

        data['env'] = [x.tolist() for x in [ts,env_p, env_m]]

        self.plot_fft(axes,xs=self.x_data,ys=self.y_data)

        json.dump(data,open('dumper.json','w'))

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
        axes.legend(loc='best',framealpha=0.4)
        #axes.plot(x_data_fit, self.envelope(x_data_fit,*self.fit_parameters), '--', lw=3, label=self.compose_label())



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
        axes.legend(loc='center',framealpha=0.4)


class T1Fit(DataFit):
    x_row = 'T pulse' #5+2
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
        pi_time = self.get_data_from_row(row,'MW Pi pulse time')
        #pi_comp = self.get_data_from_row(row,'PI compensation')
        x_data = self.get_data_from_row(row,self.x_row)
        return x_data
        #4 -> Pi time 6-> Pi compensation
        pi_half = pi_time/2. + pi_comp
        pi = pi_time + pi_comp
        echo_correction = pi_half*2. + pi #pi_time
        return x_data*2.+echo_correction
        #4 -> Pi time 6-> Pi compensation
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
        axes.legend(loc='best',framealpha=0.4)

class polarization(DataFit):
    #x_row = 'T pulse' #3+2
    x_row = 'Delay to ref.col.'
    counts_type = 2
    fit_params_description = ['intensity',
                              'T_init',
                              'level',
                              ]
    label_fit_params = [0,1,2]
    label_data_params = [0,1] # frequency, power
    data_format = 'k-+'
    fit_format = 'r-'

    def initial_guess(self,x_data,y_data): #Must be defined by user
        return [0.05,5000,0.0]
    def fit_fun(self,t,intensity, T2,level): #Must be defined by user
        return level+(intensity*np.exp(-(t/T2)))
    def x_fun(self, row):
        return self.get_data_from_row(row,self.x_row)

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
        signal = np.array(self.all_data[:,-1],dtype=float)
        reference = np.array(self.all_data[:,-2],dtype=float)
        norm = np.mean((signal+reference)/2.)

        signal/=norm
        reference/=norm
        #axes.plot(self.x_data,signal,lw=2,alpha=0.5,color='g',label='Signal')
        #axes.plot(self.x_data,reference,lw=2,alpha=0.5,color='r',label='Reference')
        axes.set_xlabel(self.plot_x_label())
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='best',framealpha=0.4)

class XY_mapfit():
    data_fit = ESRFit2
    map_fit_parameter = 2 #2
    plot_fits = True
    subfit_plot_params = {}
    x_column_name = 'X'
    y_column_name = 'Y'
    x_axis_label  = 'X, Volt(~100um per Volt)'
    y_axis_label  = 'Y, Volt(~100um per Volt)'
    #um_per_volts = 6.6
    def __init__(self, data, headers=[].copy()):
        self.data_headers = headers
        self.first_row = data[0]
        self.all_data = np.array(data)
        self.x_idx = headers.index(self.x_column_name)
        self.y_idx = headers.index(self.y_column_name)
        self.Xs = np.unique(self.all_data[:,self.x_idx])
        self.Ys = np.unique(self.all_data[:,self.y_idx])
        self.xxs,self.yys = np.meshgrid(self.Xs,self.Ys)

        self.subfits = []
        map_data = []
        if isinstance(self.map_fit_parameter,str):
            fit_param_idx = self.data_fit.label_fit_params.index(self.map_fit_parameter)
        else:
            fit_param_idx = self.map_fit_parameter
        for x,y in zip(self.xxs.flatten(),self.yys.flatten()):
            cur_fit = self.data_fit(self.select_data_by_XY(x,y),self.data_headers)
            self.subfits.append(cur_fit)
            try:
                map_data.append(cur_fit.fit_parameters[fit_param_idx])
            except:
                map_data.append(np.nan)
        self.map_data = np.array(map_data).reshape(self.xxs.shape)
    def select_data_by_XY(self,x,y):
        row_idxs = np.logical_and( self.all_data[:, self.x_idx] == x, self.all_data[:, self.y_idx] == y).nonzero()[0]
        return self.all_data[row_idxs,:]
    def plot_me(self, axes, data_format=None, fit_format=None, comment=None):
        assert isinstance(axes,matplotlib.axes.Axes)
        shape = self.xxs.shape
        if shape[0] == 1:
            axes.plot(self.xxs.flatten(),self.map_data.flatten())
            axes.set_xlabel(self.x_axis_label)
            axes.set_ylabel(self.data_fit.fit_params_description[self.map_fit_parameter])
        elif shape[1] == 1:
            axes.plot(self.yys.flatten(),self.map_data.flatten())
            axes.set_xlabel(self.y_axis_label)
            axes.set_ylabel(self.data_fit.fit_params_description[self.map_fit_parameter])
        else:
            CS = axes.contour(self.xxs,self.yys,self.map_data)
            axes.clabel(CS, inline=1, fontsize=10)
            axes.set_xlabel(self.x_axis_label)
            axes.set_ylabel(self.y_axis_label)
        axes.set_title(comment)
        if self.plot_fits:

            xlen = len(self.Xs)
            ylen = len(self.Ys)
            gs = gridspec.GridSpec(ylen,xlen*2)

            fig = axes.get_figure()
            fig.set_figwidth(fig.get_figwidth()*2.)
            axes.set_position(gs[:,:xlen].get_position(fig))
            axes.set_subplotspec(gs[:,:xlen])

            for i,f in enumerate(self.subfits):
                a = fig.add_subplot(gs[ylen-1-int(i/xlen),xlen+i%xlen])

                f.plot_me(a,**self.subfit_plot_params)
                a.set_xlabel('')
                a.set_ylabel('')
                a.xaxis.set_major_formatter(plt.FormatStrFormatter('%1.0f'))
                a.yaxis.set_major_formatter(plt.FormatStrFormatter('%1.3f'))
                #a.locator_params(nbins=1)
                x = self.xxs.flatten()[i]
                y = self.yys.flatten()[i]
                fontdict={
                    'fontsize': 6,
                     'fontweight' : 'bold',
                     'verticalalignment': 'baseline',
                     'horizontalalignment': 'center'}
                a.set_title('({0:.2f},{1:.2f})'.format(x,y),fontdict=fontdict)
                a.get_xaxis().set_ticks([min(f.x_data),max(f.x_data)])
                a.get_yaxis().set_ticks([min(f.y_data),max(f.y_data)])
                for tick in a.xaxis.get_major_ticks():
                    tick.label.set_fontsize(6)
                for tick in a.yaxis.get_major_ticks():
                    tick.label.set_fontsize(6)
                a.legend().set_visible(False)


class FitSweep():
    data_fit = RabiFit
    source_fit_parameter = 6
    plot_fit_parameter = 2
    verbose = True
    def __init__(self, data, headers=[].copy()):
        self.data_headers = headers
        self.first_row = data[0]
        self.all_data = np.array(data)
        sweep_cols,sweep_names,sweep_vals = self.get_sweeps()
        if len(sweep_cols)>1:
            raise Exception('FitSweep: unsupported number of sweep parameters: {0}'.format(len(sweep_cols)))

        self.subfits = []
        self.fit_data = np.zeros([len(vals) for vals in sweep_vals])
        self.fit_data_error = np.zeros([len(vals) for vals in sweep_vals])
        self.fit_val_name = ''
        if isinstance(self.plot_fit_parameter,str):
            fit_param_idx = self.data_fit.label_fit_params.index(self.plot_fit_parameter)
        else:
            fit_param_idx = self.plot_fit_parameter
        for indexes in product(*[range(len(vals)) for vals in sweep_vals]):
            cur_sweep_vals = [vals[i] for vals,i in zip(sweep_vals,indexes)]
            try:
                cur_data = self.select_data_sweep(sweep_cols,cur_sweep_vals)
                cur_fit = self.data_fit(cur_data,self.data_headers)
                self.fit_data[tuple(indexes)] = cur_fit.fit_parameters[fit_param_idx]
                self.fit_data_error[tuple(indexes)] =cur_fit.fit_parameters_error[fit_param_idx]
                self.fit_val_name = cur_fit.fit_params_description[self.plot_fit_parameter]
            except Exception as e:
                print(e)
                self.fit_data[tuple(indexes)] = np.nan
                self.fit_data_error[tuple(indexes)] = np.nan
        self.sweep_names = sweep_names
        self.sweep_cols = sweep_cols
        self.sweep_vals = sweep_vals
        self.fit_param_idx = fit_param_idx

    def get_sweeps(self):
        sweep_cols         = []
        sweep_vals         = []
        sweep_header_names = []
        for i in range(len(self.data_headers)-2):# -2 as we avoid counts data
            cur_uni_vals = np.unique(self.all_data[:,i])
            if len(cur_uni_vals)>1 and i!=self.source_fit_parameter:
                sweep_cols.append(i)
                sweep_vals.append(cur_uni_vals)
                sweep_header_names.append(self.data_headers[i])
                if self.verbose:
                    print('FitSweep: Found sweep parameter {0}'.format(self.data_headers[i]))
        return sweep_cols,sweep_header_names,sweep_vals
    def select_data_sweep(self,sweep_cols,sweep_vals):
        current_data = self.all_data
        for col,val in zip(sweep_cols,sweep_vals):
            row_idxs = current_data[:, col] == val
            current_data = current_data[row_idxs,:]
        return current_data
    def plot_me(self, axes, data_format=None, fit_format=None, comment=None):
        assert isinstance(axes,matplotlib.axes.Axes)
        c_data = '#7f7f7f'
        c_fit = '#ff7f0e'
        assert self.fit_data.ndim==1

        x_data = self.sweep_vals[0]
        y_data = self.fit_data
        axes.plot(x_data, y_data, '+-', color = c_fit, lw=3,)
        #axes.plot(self.x_data, self.y_data, data_format)
        axes.set_xlabel(self.sweep_names[0])
        axes.set_ylabel(self.fit_val_name)
        #leg = axes.legend(loc='upper right',framealpha=0.4)
        #leg.get_frame().set_alpha(0.5)
class RabiCompositeSweep(FitSweep):
    data_fit = RabiFit1
    source_fit_parameter = 6
    plot_fit_parameter = 2
    verbose = True
    #NOT FULLY TESTED YET

class ESR_map(XY_mapfit):
    pass

class RabiFit1_no_spectrum(RabiFit1):
    def __init__(self,*args,**kwargs):
        super(RabiFit1_no_spectrum,self).__init__(*args,**kwargs)
        self.plot_power_spectrum = False
        self.plot_envelope = True
class Rabi_map(XY_mapfit):
    data_fit = RabiFit1_no_spectrum
    map_fit_parameter = 2 # Frequency
    subfit_plot_params = {}
class Rabi_map_power_freq(XY_mapfit):
    data_fit = RabiFit1_no_spectrum
    map_fit_parameter = 2 # Frequency
    subfit_plot_params = {}
    x_column_name = "MW Power"
    y_column_name = "MW Frequency"
    x_axis_label  = 'MW Power,dB '
    y_axis_label  = 'MW Frequency'
    def plot_me(self,axes, *args,**kwargs):
        super(Rabi_map_power_freq,self).plot_me(axes,*args,**kwargs)
        axes.set_yscale('log')
class RabiFitScanFreq(DataFit):
    counts_type = 1
    smooth_f = 5
    data_fit=RabiFit1
    def __init__(self,data, headers=[].copy()):
        self.data_headers = headers
        self.first_row = data[0]
        self.all_data = np.array(data)
        self.F_idx = headers.index('MW Frequency')
        self.T_idx = headers.index('T pulse')
        self.Fs = np.unique(self.all_data[:,self.F_idx])
        self.Ts = np.unique(self.all_data[:,self.T_idx])
        self.Tss,self.Fss = np.meshgrid(self.Ts,self.Fs)
        self.Signal = np.array([self.y_fun(row) for row in self.all_data])
        signal = []
        self.subfits = []
        self.Rabis = np.zeros(self.Fs.shape)
        self.Rabi_errors = np.zeros(self.Fs.shape)
        rabi_idx = self.data_fit.fit_params_description.index('Rabi frequency')
        def select_and_smooth(i):
            sw = int(self.smooth_f/2)
            selected_idx = np.arange(i-sw,i+sw+1)
            selected_idx = selected_idx[np.logical_and(selected_idx>0,selected_idx<len(self.Fs))]
            d = [self.select_data_by_freq(self.Fs[i]) for i in selected_idx]
            out = d[0]
            for dd in d[1:]:
                out+=dd
            return out/len(d)
        for i,f in enumerate(self.Fs):
            good_idx = self.all_data[:,self.F_idx] == f
            good_Ts  = self.all_data[good_idx,self.T_idx]
            subsignal = self.Signal[good_idx]
            signal.append(subsignal[np.argsort(good_Ts)])
            cur_data = select_and_smooth(i)
            try:
                cur_fit = self.data_fit(cur_data,self.data_headers)
                self.Rabis[i] = cur_fit.fit_parameters[rabi_idx]
                self.Rabi_errors[i] = cur_fit.fit_parameters_error[rabi_idx]
            except:
                cur_fit = None
                self.Rabis[i] = np.nan
                self.Rabi_errors[i] = np.nan
            self.subfits.append(cur_fit)

        self.map_data = np.array(signal)
        filt = np.ones((self.smooth_f,1))
        d = s.convolve(self.map_data,filt,'same')
        d /= s.convolve(np.ones((self.map_data.shape[0],1)),filt,'same')
        self.map_data = d
    def plot_me(self, axes, data_format=None, fit_format=None, comment=None):
        assert isinstance(axes,matplotlib.axes.Axes)
        im = axes.pcolormesh(self.Ts,self.Fs,self.map_data)
        plt.gcf().colorbar(im)
        axes.set_ylabel('F, Mhz')
        axes.set_xlabel('T, ns')
        axes.set_title(comment)
        text_x = self.Ts[int(len(self.Ts)*0.7)]
        for F,rabi,rabi_err in zip(self.Fs,self.Rabis,self.Rabi_errors):
            axes.text(x=text_x,y=F,s='{0:.2f} ({1:.3f}) MHz'.format(rabi,rabi_err))
        sp_plot = plt.axes([0.4,0.2,0.3,0.3])
        sp_plot.set_xlabel('F, Mhz')
        sp_plot.set_ylabel('$\Omega_R/2\pi$')
        sp_plot.set_alpha(0.9)
        sp_plot.plot(self.Fs,self.Rabis)
        sp_plot.fill_between(self.Fs,self.Rabis-self.Rabi_errors,self.Rabis+self.Rabi_errors,alpha=0.5)
    def select_data_by_freq(self,freq):
        row_idxs = self.all_data[:, self.F_idx] == freq
        return self.all_data[row_idxs,:]
