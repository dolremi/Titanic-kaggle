# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:52:32 2016

@author: jiaxu
"""
import pandas as pd
import numpy as np
from dataset import DataUtilityBase

class TitanicData:
   """
   TitanicData class will read in the training and test dataset from Titanic
                     output the prediction result for submission
                     handle missing features
                     do the predictions
   """
   def __init__(self, train_name=None, test_name=None):
       if train_name is None:
           train_name = "../../data/original/train.csv"
       if test_name is None:
           test_name = "../../data/original/test.csv"
       self.__train_data = pd.read_csv(train_name, dtype={"Age": np.float64})
       self.__test_data = pd.read_csv(test_name, dtype={"Age":np.float64}) 
       self.__train_utility = DataUtilityBase(self.__train_data, "training")
       self.__test_utility = DataUtilityBase(self.__test_data, "test")
    
   def output(self, train_out, test_out):
       if train_out is None:
           train_out = "../../data/modified/train.csv"
       if test_out is None:
           test_out = "../../data/modified/test.csv"
       self.__train_data.to_csv(train_out)
       self.__test_data.to_csv(test_out)
    
   def show_features(self, name="train"):
       if name == "train":
           self.__train_utility.show_features()
       else:
           self.__test_utility.show_features()
           
   def clean(self):
       self.__train_data["Embarked"].fillna(
                     self.__train_data["Embarked"].value_counts().id_max(), 
                                            inplace = True)
       self.__test_data["Fare"].fillna(self.__test_data["Fare"].median(), 
                                            inplace = True)
       self.__train_utility.trans_embarked()
       self.__test_utility.trans_embarked()
   
   def gen_type(self):
       self.__train_utility.type_gen()
       self.__test_utility.type_gen()
       
   def gen_family(self):
       self.__train_utility.family_gen()
       self.__test_utility.family_gen()
    
   def gen_title_and_last(self):
       self.__train_utility.title_and_last_gen()
       self.__test_utility.title_and_last_gen()
       
   
