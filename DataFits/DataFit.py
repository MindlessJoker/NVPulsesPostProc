__author__ = 'Vladimir'

import numpy as np
from scipy import optimize
from pylab import ravel
from random import randrange

def rand_color():
    colors = 'rbgkyc'
    return colors[randrange(0, len(colors))]

class DataFit:
    x_row = 0  # Must be defined by user
    counts_type = 1  # 0->absolute, 1->relative, 2->contrast
    fit_params_description = []  # Must be defined by user
    label_fit_params = []  # Should be defined by user
    label_data_params = []
    fit_format = None
    data_format = None
    def get_data_from_row(self,row,data_idx):
        if isinstance(data_idx,str):
            try:
                idx = self.data_headers.index(data_idx)
            except ValueError:
                raise ValueError('"{0}" does not present in data headers({1})'.format(data_idx,self.data_headers))
        else:
            idx = data_idx
        return row[idx]
    def __init__(self, data, headers=[].copy()):
        self.data_headers = headers
        self.first_row = data[0]
        self.all_data = np.array(data)
        self.y_data = np.array(list(map(self.y_fun, data)))
        self.x_data = np.array(list(map(self.x_fun, data)))
        self.try_fit(self.x_data, self.y_data)


    def x_fun(self, row):
        return self.get_data_from_row(row,self.x_row)
        # if isinstance(self.x_row,str):
        #     idx = self.data_headers.index(self.x_row)
        # else:
        #     idx = self.x_row
        # return row[idx]

    def y_fun(self, row):
        if self.counts_type == 0:
            return row[-1]
        if self.counts_type == 1:
            return row[-1] / row[-2] if row[-2] != 0.0 else -1.0
        if self.counts_type == 2:
            return (row[-1] - row[-2]) / (row[-1] + row[-2])*2. if (row[-1] + row[-2]) != 0.0 else -1.0

    def initial_guess(self, x_data, y_data):  # Must be defined by user
        return [0]

    def fit_fun(self, x, *params):  # Must be defined by user
        return 0

    def build_fit(self, *params):
        if isinstance(params, np.ndarray):
            params = params.tolist()
        return np.vectorize(lambda t: self.fit_fun(t, *params))
    def label_additional_info(self):
        """
        :return: list of strings, that will be appended to label info data
        """
        return []
    def parameter_scoring(self,*params):
        return 0
    def try_fit(self, x_data, y_data):
        error = 0
        params = np.array(self.initial_guess(x_data, y_data))
        errorf = lambda p: ravel(self.build_fit(*p)(x_data) - y_data) + np.repeat([self.parameter_scoring(*p)],len(y_data))
        pfit, pcov, infodict, errmsg, success = optimize.leastsq(errorf, params, maxfev=1000,full_output=True)
        self.fit_parameters = pfit
        self.fit_error = errorf(pfit) / float(len(x_data))
        if pcov is not None:
            sum_squares = (errorf(pfit)**2).sum() / float(len(x_data)-len(pfit))
            self.fit_parameters_error = np.diag(abs(pcov*sum_squares))**0.5
        else:
            self.fit_parameters_error = np.zeros(pfit.shape)

        # self.fit_success = success!=
        #return success

    def fitted_data(self, n_points=1000):
        fit_f = self.build_fit(*(self.fit_parameters))
        #self.y_data /= self.fit_parameters[0]
        x_min = np.min(self.x_data)
        x_max = np.max(self.x_data)
        x_delta = (x_max - x_min) / n_points
        fit_x = np.arange(x_min, x_max + x_delta, x_delta)
        fit_y = fit_f(fit_x) if len(fit_x) > 0 else []
        return fit_x, fit_y

    def compose_label(self):
        label_data = []

        def get_data(idxs, headers, values, data_type,value_errors=None):
            for l in idxs:
                if isinstance(l,str):
                    try:
                        l = headers.index(l)
                    except ValueError:
                        print('Data header is not found when building info label: {0} in {1}'.format(l,headers))
                        continue
                try:
                    if value_errors is not None:
                        label_data.append(
                            '{0} = {1:.2f} ({2:.2f})'.format(headers[l], values[l],value_errors[l])
                        )
                    else:
                        label_data.append(
                            '{0} = {1:.2f}'.format(headers[l], values[l])
                        )
                except IndexError:
                    print('No data for {0} {1} parameter'.format(l, data_type))
                    continue

        get_data(self.label_data_params, self.data_headers, self.first_row, 'data')
        get_data(self.label_fit_params, self.fit_params_description, self.fit_parameters, 'fit',self.fit_parameters_error)
        label_data += self.label_additional_info()
        return '\n'.join(label_data)

    def plot_x_label(self):
        if isinstance(self.x_row,str):
            idx = self.data_headers.index(self.x_row)
        else:
            idx = self.x_row
        try:
            return self.data_headers[idx]
        except IndexError:
            return ''

    def plot_y_label(self):
        if self.counts_type == 0:
            return 'Absolute counts'
        if self.counts_type == 1:
            return 'Relative counts'
        if self.counts_type == 2:
            return 'Contrast'

    def plot_me(self, axes, data_format=None, fit_format=None,comment=None):
        c = rand_color()
        data_format = self.data_format if data_format is None else data_format
        fit_format = self.fit_format if fit_format is None else fit_format
        data_format = c + '+-' if data_format is None else data_format
        fit_format = c + '-' if fit_format is None else fit_format
        axes.plot(self.x_data, self.y_data, data_format)
        x_data_fit, y_data_fit = self.fitted_data()
        axes.plot(x_data_fit, y_data_fit, fit_format, lw=3, label=self.compose_label())
        axes.plot(self.x_data, self.y_data, data_format)

        axes.set_xlabel(self.plot_x_label())
        axes.set_ylabel(self.plot_y_label())
        axes.legend(loc='center')

    def plotpulses(self,axes, lengths): # length in Pi
        fit_f = self.build_fit(*(self.fit_parameters))
        compensation = self.fit_parameters[-1]
        pi_pulse_length = self.fit_parameters[-2]
        for pulse_length in lengths:
            xs = pulse_length *pi_pulse_length  - compensation
            ys = fit_f(xs)
            #axes.plot(xs,ys, 'ro', ms = 10)


