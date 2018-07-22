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
    #RabiFit3sin)
    data_fit_funcs = {
        "Rabi oscillation" : DataFitBySweepSelector( [
                    ("T pulse",RabiFit1),
                    ("MW Frequency",ESRFit_pulsed),
                    #(["T pulse","MW Power"],RabiCompositeSweep),
                    (["T pulse","MW Power"],Rabi_map_power_freq),
                    (["T pulse","MW Frequency"],RabiFitScanFreq),
                    (['X','Y',"T pulse"],Rabi_map),
                    (['X',"T pulse"],Rabi_map),
                    (['Y',"T pulse"],Rabi_map)
        ]),
        "Pi pulse check"   : DataFitBySweepSelector( [ ("T pulse",PiPulseFit),("MW Frequency",ESRFit) ]),
        "Excitation-collection align": ExcitationCollectionAlignmentFit,
        "ESR": DataFitBySweepSelector( [
                ("Delay to ref.col.",polarization),
                ("MW Frequency",ESRFit),
                (['X','Y',"MW Frequency"],ESR_map) ]
        ),
        "Spin echo": EchoFit4,
        "Dynamic Scheme": DynamicScheme,
        "T1":T1Fit,
        "Ramsey\n":RamseyFit
    }
    modulation_scheme_alias = {
        'Rabi oscillation (unbalanced)': 'Rabi oscillation'
    }
    def __init__(self,data_dict):
        self.data = data_dict
        mod_scheme = self.data["modulation_scheme"]
        if mod_scheme not in self.data_fit_funcs.keys():
            mod_scheme = self.modulation_scheme_alias[mod_scheme]
        fit = self.data_fit_funcs.get(mod_scheme,DataFit)
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
