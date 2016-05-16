# Kaggle Titanic Challenge

The Goal of this project is to predict the survival of the person given the information.

## Pre-requests
- Python 2.7
- [Pandas](http://pandas.pydata.org/): data analysis
- [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/): data visualizatoin
- [scikit-learn](http://scikit-learn.org/): random forest
- [Keras](http://keras.io/): neural networks
- [Theano](https://github.com/Theano/Theano): tenser operations, used as Keras's backend

## Quick start
- See `visualize/Data_Explore_Jia.ipynb` for explorations over data and features.
- Add `<repo path>` to your `PYTHONPATH` if you want to import this as package.

## What's new
- Restructured as package format. 
Run 
```
cd src/test
python testdata.py
```
to see if `data` submodule works.

## Structure
File structure is like this:
```
./
|- titanic/ # all reuseable sources as a package
  |- __init__.py
  |- data/ # modules about data parsing and processing
    |- __init__.py
    |- dataset.py 
    ...
  |- nn/ # modules about NN
    |- __init__.py
    |- model.py
    ...
|- data/ # dataset and processed data
  |- original/
    |- xxx.csv
    ...
|- visualize
  |- xxx.ipynb
|- README.md
...
```
