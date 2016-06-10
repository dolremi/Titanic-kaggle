# -*- coding: utf-8 -*-
"""
MissingFeatures module will handle missing features in the dataset including 
imputing or split data into missing and non-missing.
"""

import pandas as pd
import numpy as np
from pandas import Series
from sklearn.preprocessing import Imputer


class MissingFeatures:
    """Handler class for missing features
    
    Parameters
    ----------
    dataset : Pandas dataframe that will be imputed
    
    """
    def __init__(self,
                 column_names = None,
                 missing_values = "NaN", 
                 strategy = "mean",
                 axis = 0):
        self.columns = column_names
        self.missing_value = missing_values
        self.strategy = strategy
        self.axis = axis
        
    def _check_data(self, dataset):
        if dataset is None:
            raise ValueError("The input dataset cannot be empty.")

        if not (set(self.columns) < set(dataset.columns)):
            raise ValueError("Not all columns {0} are in"
                             "the columns of the dataset"
                             " which are: {1}".format(self.columns,       
                             dataset.columns))
                             
        if self.axis not in [0, 1]:
            raise ValueError("Only axis 0 or 1 can be imputed"
                             " got axis={0}".format(self.axis))
                                                              
    def _check_impute(self):
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("The availabe strategeis are: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))
    def _check_column(self, column, dataset):
        if column not in dataset.columns:
            raise ValueError("{0} is not in the columns of the dataset"
                              "which are:m{1}".format(column, dataset.columns))
            
    def split_models(self, dataset, column):
        self._check_column(dataset,column)           
        nonmissing = dataset[pd.notnull(dataset[column])]
        missing = dataset[pd.isnull(dataset[column])].drop(column, axis = 1)
        return (missing, nonmissing)
        
    def drop_missing(self, dataset):
        self._check_data()    
        dataset.dropna(self.column, self.axis)  
            
    def impute_regular(self, train, test = None):
        self._check_data(train)
        for col in self.columns:
            if train[col].dtype == np.int64:
                self.strategy = "median"
            elif train[col].dtype == np.float64:
                self.strategy = "mean"
            else:
                self.strategy = "most_frequent"
            imr = Imputer(self.missing_value, self.strategy, self.axis)
            imr = imr.fit(train[col])
            train[col] = Series(imr.transform(train[col]).T.ravel())
            if test is not None:
                test[col] = Series(imr.transform(test[col]).T.ravel())
        return (train, test)
        
