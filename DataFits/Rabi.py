import numpy as np
from matplotlib import pyplot as plt

from DataFits import DataFit
from DataFits.utils import get_power_spectrum
from .utils import get_power_spectrum

class RabiFit1(DataFit): #single
    max_frequency = 11.2

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
        super(RabiFit1,self).__init__(*args,**kwargs)
        self.plot_power_spectrum = True
        self.plot_envelope = True
    def initial_guess(self,x_data,y_data): #Must be defined by user
        level = max(y_data)
        fs,pwrs = get_power_spectrum(x_data*1e-3,y_data,df=0.1,maxF=self.max_frequency)
        f = fs[np.argmax(pwrs[1:])+1] #exclude 0 freq
        #print('Rabi fit: guessed F={0}'.format(f))
        return [level,0.3, f, 6000.0,0.0]
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
