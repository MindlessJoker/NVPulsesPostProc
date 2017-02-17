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
enable_date_filter = True
antenna_stats = open('antenna_stats2.csv','w')
date_to_make = '16-02-2017'
antenna_stats.write('\t'.join(['Date','Comment','Ant','Rabi freq','Contrast','Power,dB','Rabi freq @ 40dB','Resonant frequency'])+'\n')
for i,f in enumerate(files):
    if enable_date_filter:
        if f.endswith('.json'):
            date_cur,time_cur = f.split('_')
            time_cur = time_cur[0:-5]
            if date_cur != date_to_make:
                continue
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
            resonant_freq = data["data"][0][0]
            out_d = [data["date"],comment,antenna,freq,intensity,power,freq_40,resonant_freq]
            out_string = '\t'.join(map(str,out_d))
            antenna_stats.write(out_string+'\n')
            print(antenna, freq_40)

    except :
        print('Exception!')

antenna_stats.close()