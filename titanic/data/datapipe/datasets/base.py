# -*- coding: utf-8 -*-
"""
Base IO code for sample datasets
"""

from os.path import dirname
from os.path import join
import pandas as pd


def load_titanic_train():
    
    module_path = dirname(__file__)
    data_file = pd.read_csv(join(module_path,"data","titanic","train.csv"))
    return data_file
    
    
def load_titanic_test():
    module_path = dirname(__file__)
    data_file = pd.read_csv(join(module_path,"data","titanic","test.csv"))
    return data_file