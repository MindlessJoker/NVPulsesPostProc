__author__ = 'Vladimir'

import PulsesDataProcessor
import ESR,RABI
from matplotlib import pyplot as plt
import numpy as np
import os
import json
DataDir = "D:\\Data\\!pulses data"
files = os.listdir(DataDir)
#rint(files)

antenna_stats = open('antenna_stats.csv','w')
antenna_stats.write('\t'.join(['Date','Comment','Ant','Rabi freq','Contrast','Power,dB','Rabi freq @ 40dB'])+'\n')
for i,f in enumerate(files):
    try:
        data_file = open(os.path.join(DataDir,f),'r')
        data = json.load(data_file)
        if data["modulation_scheme"] == "Rabi oscillation":

            proc = PulsesDataProcessor.PulsesDataProcessor(data)
            def rep(s):
                return s.replace('\n',' ').replace('\t',' ')
            comment = rep(data["comment"])
            antenna = rep(data["mw_antenna_type"])
            freq = proc.fit.fit_parameters[2]
            intensity = proc.fit.fit_parameters[1]
            power = data["data"][0][1]
            freq_40 = freq/10**(float(power-40)/20)
            out_d = [data["date"],comment,antenna,freq,intensity,power,freq_40]
            out_string = '\t'.join(map(str,out_d))
            antenna_stats.write(out_string+'\n')
            print(antenna, freq_40)

    except :
        print('Exception!')

antenna_stats.close()