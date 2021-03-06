from sklearn import tree
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import graphviz
import os
import numpy as np
import pandas as pd

#path for GraphViz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#load data
trainData = pd.read_csv('../data/processed/trainFeatures.csv')

#preprocess data, change string data to numerical
Yle = preprocessing.LabelEncoder()
Yle.fit(list(trainData['zoneType'].unique()))
newY = Yle.transform(Yle.classes_)
#print(newY)

class_names = Yle.classes_
print(class_names)

oldY = list(Yle.inverse_transform(newY))

#get relevant data for feature training
dataNoID = trainData.drop(['zoneType'], axis=1)
newData = pd.DataFrame()
newData['structCount'] = dataNoID['structCount']
newData['structRatio'] = dataNoID['structRatio']
newData['recScore'] = dataNoID['recScore']
X_train = newData
feature_names = list(newData)
print(X_train.head())

#get target data
Y_train = Yle.transform(trainData['zoneType'])

#get test data
testData = pd.read_csv('../data/processed/testFeatures.csv')
dataNoID = testData.drop(['zoneType'], axis=1)
newTestData = pd.DataFrame()
newTestData['structCount'] = dataNoID['structCount']
newTestData['structRatio'] = dataNoID['structRatio']
newTestData['recScore'] = dataNoID['recScore']
X_test = newTestData
y_test = Yle.transform(testData['zoneType'])

print(X_test.head())

#build decision tree
clf=RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, Y_train)
y_pred = clf.fit(X_train, Y_train).predict(X_test)
# dot_data = tree.export_graphviz(
# 	clf,
# 	feature_names=feature_names,
# 	class_names=class_names,
# 	filled=True,
# 	rounded=True,
# 	special_characters=True,  
# 	out_file=None)

# graph = graphviz.Source(dot_data)
# graph.render('test')

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