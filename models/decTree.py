from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import os
import numpy as np
import pandas as pd

#path for GraphViz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#load data
data = pd.read_csv('../data/processed/train.csv')

#get relevant data for feature training
dataNoID = data.drop(['zoneID', 'zoneType'], axis=1)
X = pd.concat( [data['zoneArea'], data['StructureCount']], axis=1).values

#get target data
Y = data['zoneType'].values

print(type(X))
print(type(Y))

#build decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None)

graph = graphviz.Source(dot_data)
graph.render('test')
