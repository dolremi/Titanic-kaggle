# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:59:57 2016

@author: jiaxu
"""

import re
from Dataset import DataUtilityBase

class TitanicUtility(DataUtilityBase):
    
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
        
    