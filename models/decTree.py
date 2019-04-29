from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import preprocessing
import graphviz
import os
import numpy as np
import pandas as pd

#path for GraphViz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#load data
data = pd.read_csv('../data/processed/train.csv')

#preprocess data, change string data to numerical
Yle = preprocessing.LabelEncoder()
Yle.fit(list(data['zoneType'].unique()))
newY = Yle.transform(Yle.classes_)
print(newY)

oldY = list(Yle.inverse_transform(newY))

print(oldY)

#get relevant data for feature training
dataNoID = data.drop(['zoneID', 'zoneType'], axis=1)
X = data.values



#get target data
Y = data['zoneType']

print(type(X))
print(type(Y))

#build decision tree
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# dot_data = tree.export_graphviz(clf, out_file=None)

# graph = graphviz.Source(dot_data)
# graph.render('test')
