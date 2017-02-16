import PulsesDataProcessor

__author__ = 'Vadim'


import ESR,RABI
from matplotlib import pyplot as plt
import numpy as np
import os
import json
DataDir = "D:\\Data\\!pulses data"
files = os.listdir(DataDir)
#rint(files)

class data():
    def __init__(self,filelist):
        self.files = filelist
    def slice_by_date(self,date):
        files = []
        for f in self.files:
            if f.endswith('.json'):
                date_cur = f.split('_')[0]
                #print(date_cur)
                if date_cur == date:
                    files.append(f)
        return files

    def slice_by_date_crop_by_time(self,date,time1 = '00-00-00',time2 = '23-59-59'):
        file = []
        for f in self.files:
            if f.endswith('.json'):
                date_cur,time_cur = f.split('_')
                time_cur = time_cur[0:-5]
                if date_cur == date:
                    print(1)
                    if len(time_cur) == 7:
                        time_cur = '0' + time_cur
                        #print('costil',time_cur)
                    if time_cur > time1:
                        print(2)
                        if time_cur < time2:
                            print(3)
                            file.append(f)
        return file
newdata = data(files)

#####################################
# USER interface
#####################################

files_by_date = newdata.slice_by_date_crop_by_time('16-02-2017',time1 = '00-00-00')

freqs = []
powers = []
print(files_by_date)

for i,f in enumerate(files_by_date):
    print(f)
    picturename = f[0:-5]+'_.png'
    if not os.path.exists(picturename):
        data_file = open(os.path.join(DataDir,f),'r')
        data = json.load(data_file)
        proc = PulsesDataProcessor.PulsesDataProcessor(data)
        print(data['comment'])
        ax = plt.subplot(111)
        proc.plot(ax)
        plt.legend(loc='best')
        plt.savefig(f[:-5]+'_.png')
        plt.close()
    else:
        print('skip already exists')

# ax = plt.subplot(111)
#
# for i,f in enumerate(files_by_date):
#     print(f)
#     fullfile = os.path.join(DataDir,f)
#     newfile = RABI.rabi_data(fullfile)
#     newfile.plot_normalized_data(ax,fit=True, legend = True,loc=0,plot_id=i)
# ax.set_xlabel('T pulse, ns')
#
# plt.show()

# for i,f in enumerate(files_by_date):
#     print(f)
#     fullfile = os.path.join(DataDir,f)
#     try:
#         newfile = RABI.rabi_data(fullfile)
#     except :
#         continue
#     ax = plt.subplot(111)
#     newfile.plot_normalized_data(ax,fit=True, legend = True,loc=0,plot_id=i)
#     ax.set_xlabel('T pulse, ns')
#     plt.savefig(f[:-5]+'.png')
#     plt.close()
#     freqs.append(abs(newfile.params[2]*1000))
#     powers.append(np.power(10,(newfile.data[0][1]-30)/10/2))
# ax = plt.subplot(111)
# ax.plot(powers,freqs,'+')
# ax.set_xlabel('SQRT POWER, W^0.5')
# ax.set_ylabel('Rabi frequency, MHz')
# plt.savefig('freq_power.png')
# plt.close()