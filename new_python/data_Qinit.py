"""
To add to clawpack.geoclaw.data,
to augment and replace current QinitData class
"""

from __future__ import print_function
from __future__ import absolute_import
import os

import clawpack.clawutil.data
import six
from six.moves import range
import numpy as np


class ForceDry(clawpack.clawutil.data.ClawData):
    
    def __init__(self):
        r"""
        A single force_dry array and associated data
        """
        
        super(ForceDry,self).__init__()
        self.add_attribute('tend',None)
        self.add_attribute('fname','')


class QinitData(clawpack.clawutil.data.ClawData):

    def __init__(self):

        super(QinitData,self).__init__()
        
        # Qinit data
        self.add_attribute('qinit_type',0)
        self.add_attribute('qinitfiles',[])   
        self.add_attribute('variable_eta_init',False)   
        self.add_attribute('force_dry_list',[])   
        self.add_attribute('num_force_dry',0)

    def write(self,data_source='setrun.py', out_file='qinit.data'):

        print('+++ qinit.variable_eta_init = ', self.variable_eta_init)
        # Initial perturbation
        self.open_data_file(out_file, data_source)
        self.data_write('qinit_type')

        # Perturbation requested
        if self.qinit_type == 0:
            pass
        else:
            # Check to see if each qinit file is present and then write the data
            for tfile in self.qinitfiles:
                # if path is relative in setrun, assume it's relative to the
                # same directory that out_file comes from
                fname = os.path.abspath(os.path.join(os.path.dirname(out_file),tfile[-1]))
                self._out_file.write("\n'%s' \n" % fname)
                self._out_file.write("%3i %3i \n" % tuple(tfile[:-1]))
        # else:
        #     raise ValueError("Invalid qinit_type parameter %s." % self.qinit_type)


        self.data_write('variable_eta_init')

        self.num_force_dry = len(self.force_dry_list)
        self.data_write('num_force_dry')

        for force_dry in self.force_dry_list:
            
            # if path is relative in setrun, assume it's relative to the
            # same directory that out_file comes from
            fname = os.path.abspath(os.path.join(os.path.dirname(out_file),\
                    force_dry.fname))
            self._out_file.write("\n'%s' \n" % fname)
            self._out_file.write("%.3f \n" % force_dry.tend)

    
        self.close_data_file()


