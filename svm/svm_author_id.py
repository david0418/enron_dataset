#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
import os
from time import time
sys.path.append("/Users/davidding/Documents/GitHub/machine_learning/ud120-projects-master/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC()
clf.fit(features_train, labels_train)
SVC(kernel='linear')
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print accuracy
#########################################################
