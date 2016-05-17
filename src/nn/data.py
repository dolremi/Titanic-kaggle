# -*- coding: utf-8 -*-
"""
@author: jmzhao
"""

import pandas as pd
import numpy as np
import logging

class Map :
    def __init__(self, names=None, from_names=None, to_names=None, func=None) :
        self.from_names = None
        self.to_names = None
        if names is not None :
            self.from_names = names
            self.to_names = names
        if from_names is not None :
            self.from_names = from_names
        if to_names is not None :
            self.to_names = to_names
        self.func = func
        assert all(x is not None for x in (self.from_names, self.to_names, self.func))
        assert len(self.from_names) > 0
        assert len(self.to_names) > 0
        
    def __call__(self, data) :
        l_outputs = list()
        from_data = [data[col] for col in self.from_names]
        shapes = [d.shape for d in from_data]
        if not all(x[0] == shapes[0][0] for x in shapes) :
            logging.warn("Map: Not all source columns are with the length.\n" 
                + "\n".join("%s: %s"%(col, shape) 
                    for col, shape in zip(self.from_names, shapes)))
        for inputs in zip(*from_data) :
            outputs = self.func(*inputs)
            l_outputs.append(outputs)
        to_data = dict()
        for i, col in enumerate(self.to_names) :
            to_data[col] = [x[i] for x in l_outputs]
        return to_data

def identity(*args) :
    return args
    
class Select (Map) :
    def __init__(self, names) :
        super().__init__(names=names, func=identity)
        
def gen_naive_embedding_info(name, possible_values, others=False) :
    feat_map = dict((v, i) for i, v in enumerate(possible_values))
    zeros = [0 for _ in possible_values]
    default_index = None
    features = [name+'_'+str(v) for v in possible_values]
    if others : 
        features.append(name+'_'+'others')
        zeros.append(0)
        default_index = len(zeros) - 1
    def converter(x) :
        outputs = list(zeros)
        outputs[feat_map.get(x, default_index)] = 1
        return outputs
    info = {
        'from_names' : [name],
        'to_names' : features,
        'func' : converter,
    }
    return features, info
