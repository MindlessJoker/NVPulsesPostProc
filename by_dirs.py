__author__ = 'Alexey'

import os
import shutil
for f in os.listdir('.\\'):
    if f.endswith('.png'):
        dirname = f.split('_')[0]
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        shutil.copyfile(f,os.path.join(dirname,f))