# -*- coding: utf-8 -*-
"""
Dataset module to create features from the raw data set, clean the data, handle
the missing data in the feature and filter the important features in regard 
with the prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

class DataUtilityBase:
    """
    The base utility class do the basic utlity jobs includes showing all the
    basic information of the features, handle the category feature for the 
    prediction, impute the missing features via the median/mode or even via 
    machine learning algorithm, show the importance of the features.
    """
    def __init__(self, dataset, name):
        self.__data = dataset
        self.__name = name
    
    def show_all(self):
        print "Here is the feature list of %s dataset:" %self.__name
        print self.__data.columns.values    
    
    def show_info(self):
        print "Here is some basic information of %s dataset:" %self.__name
        print "Some sample datasets: "
        print self.__data.head()
        print "-"*30
        print "The data type : "
        print self.__data.dtypes
        print "-"*30
        print "The general information of the dataset: "
        print self.__data.describe()
        
    def show_feature(self, feature):
        print "Details of %s as follow:" %feature 
        if feature in self.__data.columns.values:
            print self.data[feature].describe()
       
    def handle_category(self, feature, add_na=False):
        if feature in self.__data.columns.values:
            dummy_frame = pd.get_dummies(self.__data[feature],dummy_na=add_na)
            self.__data = self.__data.join(dummy_frame)
        else:
            print "%s is not in feature list!" %feature
            print "-"*30
            self.show_all()
                
    def show_missing(self):
        print "Feature: No. of missing features"
        for feature in self.__data.columns.values:
            print feature + ": " +self.__data[feature].isnull().sum()
            
    def impute_miss_default(self, feature, average = None):
        if self.__data[feature].dtype == "category":
            self.__data[feature].fillna(self.__data[feature].mode())
        elif average == None:
            self.__data[feature].fillna(self.__data[feature].median())
        else:
            self.__data[feature].fillna(self.__data[feature].mean())
            
    def get_important_features(self, outcome, feature_list = None):
        self._pretrain(outcome, feature_list)
        forest = ExtraTreesClassifier(n_estimators = 250,
                                      random_state = 0)
        forest.fit(self.__x, self.__y)
        if feature_list == None:
            feature_list = self.__x.columns.values
            
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]
        features = []

         # Print the feature ranking
        print("Feature ranking:")
        for f in range(self.__x.shape[1]):
             print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))
             features.append(feature_list[indices[f]])

         # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.__x.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.__x.shape[1]), features)
        plt.xlim([-1, self.__x.shape[1]])
        plt.show()

    def _pretrain(self, predict_result, feature_list = None ):
        if predict_result in self.__data.columns.values:
            if feature_list == None:
                self.__x = self.__data.drop(predict_result,1)
            else:
                self.__x = self.__data[feature_list]
            self.__y = self.__data[predict_result]
        else:
            self.__x = self.__data        


            
   