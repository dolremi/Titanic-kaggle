# -*- coding: utf-8 -*-
"""
CatFeatures module will handle all the categorical features

@author: jiaxu
"""
import numpy as np
class CatFeatures:
    def __init__(self, mapping):
        self.mapping = mapping
        
    def map_ordinal(self, dataset, column):
        dataset[column] = dataset[column].map(mapping)