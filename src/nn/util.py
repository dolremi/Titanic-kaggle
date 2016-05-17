# -*- coding: utf-8 -*-
"""
@author: jmzhao
"""

# save model
def save_model(model, model_filename) :
    print(model.to_json(), file=open(model_filename+'.json', 'w'))
    model.save_weights(model_filename+'.weight.h5')

