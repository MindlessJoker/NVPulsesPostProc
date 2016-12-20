import PulsesDataProcessor

__author__ = 'Vadim'


import ESR,RABI
from matplotlib import pyplot as plt
import numpy as np
import os
import json
DataDir = "D:\\Data\\!pulses data"
#rint(files)

import sys

        data_file = open(os.path.join(DataDir,f),'r')
        data = json.load(data_file)