__author__ = 'Vladimir'

from DataFits import *

def get_sweep_parameter(data):
    all_params = set(data["data_headers"])
    constant_params = set([e["name"] for e in data["constant_parameters"]])
    sweep_params = all_params.difference(constant_params).difference({'Reference counts', 'Signal counts'})
    return sweep_params

class DataFitBySweepSelector():
    def __init__(self,data_fits):
        """
        :param data_fits: list of [[list of sweep params],DataFit]
        """
        self.data_fits = [(set(df[0] if isinstance(df[0],(list,tuple)) else [df[0]]), df[1]) for df in data_fits]
    def get_fit(self,data):
        """
        :param data: whole data for measuremnt
        :return: DataFit class
        """
        sweep_params = get_sweep_parameter(data)
        for df in self.data_fits:
            if df[0] == sweep_params:
                print(sweep_params,df[0])
                return df[1]
        return DataFit
class PulsesDataProcessor:
    data_fit_funcs = {
        "Rabi oscillation" : DataFitBySweepSelector( [ ("T pulse",RabiFit1),("MW Frequency",ESRFit) ]),
        "Pi pulse check"   : DataFitBySweepSelector( [ ("T pulse",PiPulseFit),("MW Frequency",ESRFit) ]),
        "Excitation-collection align": ExcitationCollectionAlignmentFit,
        "ESR": DataFitBySweepSelector( [ ("Delay to ref.col.",polarization),("MW Frequency",ESRFit) ] ),
        "Spin echo": EchoFit4,
        "Dynamic Scheme": DynamicScheme,
        "T1":T1Fit,
        "Ramsey\n":RamseyFit
    }
    def __init__(self,data_dict):
        self.data = data_dict
        fit = self.data_fit_funcs.get(self.data["modulation_scheme"],DataFit)
        if isinstance(fit,DataFitBySweepSelector):
            fit = fit.get_fit(data_dict)
        if fit == DataFit:
            print('Unknown modulation scheme: \'{0}\''.format(self.data["modulation_scheme"]))

        self.fit = fit(self.data["data"],self.data["data_headers"])
        if self.data["modulation_scheme"] == "Pi pulse check":
            self.plotpulses = True
        else:
            self.plotpulses = False
    def plot(self,ax):

        if self.plotpulses:
            self.fit.plotpulses(ax,[0.5,1,1.5])
        self.fit.plot_me(ax,comment=self.data["comment"].replace('\n',' '))
