# -*- coding: utf-8 -*-
"""
Dataset module to create features from the raw data set, clean the data, handle
the missing data in the feature and filter the important features in regard 
with the prediction
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier



class utility_dataset:
    """
    The class that does all the utility stuff for the dataset
    """
    def __init__(self, dataset, name):
        self.__data = dataset
        self.__name = name
    
    def type_gen(self):
        self.__data["Type"] = self.__data[["Sex","Age"]].apply(self.get_type, axis = 1)
        self.__data = self.__data.drop("Sex",1)
    
    def family_gen(self):
        self.__data["Family"] = self.__data["Parch"] + self.__data["SibSp"]
        self.__data = self.__data.drop(["Parch", "SibSp"], 1)
        
    def title_gen(self):
        titles = self.__data["Name"].apply(self.get_title)
        self.__data["Title"] = titles
    
    def last_gen(self):
        last_name = self.__data["Name"].apply(lambda x: x.split(",")[0])
        self.__data["LastName"] = last_name
    
    def title_and_last_gen(self):
        self.last_gen()
        self.title_gen()
        self.__data = self.__data.drop("Name", 1)
        
    def get_type(self, passenger):
        sex, age = passenger
        return 'child' if age < 16 else sex
    
    def get_title(self, name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
    
    def transfer_embarked(self, embark):
        if embark == "S":
            return 1
        elif embark == "C":
            return 2
        else:
            return 3
    
    def trans_embarked(self):
        self.__data["Embarked"] = self.__data["Embarked"].apply(self.transfer_embarked)

    def show_features(self):
        print "Here is the feature list of {%0} dataset:" %self.__name
        print self.__data.columns.values.to_list()
        
    def gen_train_features(self):
        if "Survived" in self.show_features():
            self.__x = self.__data.drop("Survived",1)
            self.__y = self.__data
        else:
            self.__x = self.__data
            
    def gen_important_features(self):
        self.gen_train_features()
        forest = ExtraTreesClassifier(n_estimators = 250,
                                      random_state = 0)
        forest.fit(self.__x, self.__y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]

         # Print the feature ranking
        print("Feature ranking:")
        for f in range(self.__x.shape[1]):
             print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

         # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.__x.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.__x.shape[1]), indices)
        plt.xlim([-1, self.__x.shape[1]])
        plt.show()
        

class data_set:
   """
   data_set class will handle most of the work
   """
   def __init__(self, train_name, test_name):
       if train_name is None:
           train_name = "../../data/original/train.csv"
       if test_name is None:
           test_name = "../../data/original/test.csv"
       self.__train_data = pd.read_csv(train_name)
       self.__test_data = pd.read_csv(test_name) 
       self.__train_utility = utility_dataset(self.__train_data, "training")
       self.__test_utility = utility_dataset(self.__test_data, "test")
    
   def output(self, train_out, test_out):
       if train_out is None:
           train_out = "../../data/modified/train.csv"
       if test_out is None:
           test_out = "../../data/modified/test.csv"
       self.__train_data.to_csv(train_out)
       self.__test_data.to_csv(test_out)
    
   def show_features(self, name):
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
       
   
            
   