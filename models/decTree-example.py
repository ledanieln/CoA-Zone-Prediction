from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names)

#graph = graphviz.Source(dot_data)
#graph.render("iris")

print(iris.data[0])
print(iris.target)