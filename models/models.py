from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import graphviz
import os
import numpy as np
import pandas as pd

#path for GraphViz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#load data
trainData = pd.read_csv('../data/processed/train2017-contains.csv')
testData = pd.read_csv('../data/processed/validation2017-contains.csv')

#preprocess data, change string data to numerical
Yle = preprocessing.LabelEncoder()
Yle.fit(list(trainData['zoneType'].unique()))

class_names = Yle.classes_
print(class_names)

#setup training data
X_train = trainData.drop(['zoneID', 'zoneType'], axis=1)
#X_train = pd.DataFrame(trainData['StructureCount'])
Y_train = Yle.transform(trainData['zoneType'])

#setup test data
X_test = testData.drop(['zoneID', 'zoneType'], axis=1)
#X_test = pd.DataFrame(testData['StructureCount'])
y_test = Yle.transform(testData['zoneType'])

feature_names = list(X_train)

#build decision tree
clf1 = RandomForestClassifier(n_estimators=100)
#clf = tree.DecisionTreeClassifier()
clf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=20)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = ComplementNB()
clf3 = KNeighborsClassifier(n_neighbors=4)
clf = SVC(gamma='scale', kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('rt', clf1), ('dt', clf2), ('knn', clf3)], voting='hard', weights=[1, 1, 1])
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train) 
#clf = clf.fit(X_train, Y_train)
y_pred = clf2.fit(X_train, Y_train).predict(X_test)

#Naive Bayes
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, Y_train).predict(X_test)


# dot_data = tree.export_graphviz(
# 	clf,
# 	feature_names=feature_names,
# 	class_names=class_names,
# 	filled=True,
# 	rounded=True,
# 	special_characters=True,  
# 	out_file=None)

# graph = graphviz.Source(dot_data)
# graph.render("dec")

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()