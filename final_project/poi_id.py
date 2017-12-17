#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
all_features = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments',
                'loan_advances', 'bonus', 'restricted_stock_deferred',
                'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive',
                'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person',
                'from_this_person_to_poi', 'deferred_income', 'expenses', 'restricted_stock',
                'director_fees','email_address']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print len(data_dict)# There are 146 data points

# find the number of poi and make a dictionary of poi
poi = {}
for person_name in data_dict:
    for feature in data_dict[person_name]:
        if feature not in features_list:
            features_list.append(feature)
    if data_dict[person_name]["poi"]==1:
        poi[person_name] = data_dict[person_name]

#print features_list
print len(poi)#18 poi in the dataset
print len(features_list)#21 features in the dataset

# check missing values
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person_name in data_dict:
    for feature in data_dict[person_name]:
        if data_dict[person_name][feature] == 'NaN':
            missing_values[feature] += 1

for name, value in data_dict.iteritems():
    if value['from_poi_to_this_person'] not in ['NaN',0] and\
        value['from_this_person_to_poi'] != 'NaN':
        value['poi_to_this_person_rate'] = value['from_poi_to_this_person']/\
        (value['from_poi_to_this_person']+ value['from_this_person_to_poi'])
    else:
        value['poi_to_this_person_rate'] = 0
    

# we only interest in features that does not have a lot of missing_values
for k, v in missing_values.iteritems():
    if v >= 0.7*len(data_dict):
        features_list.remove(k)

# can't covert string feature to number, email address needs to be removed
features_list.remove('email_address')

### Task 2: Remove outliers
# the outlier we found is total and the travel agency in the park
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


# univaraite feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 4)
selector.fit_transform(features, labels)
ls = zip(selector.get_support(), features_list[1:], selector.scores_)
# salary, exercised_stock_options, bonus
features_list=['poi','salary', 'exercised_stock_options', 'bonus']


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3,random_state=42)
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
### NaiveBayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred =  clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print "NaiveBayes accuracy: ", accuracy



### SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

estimators = [('reduce_dim', PCA(n_components=3)),
              ('clf', SVC())]
param_grid = dict(
         clf__C = [1e3, 5e3, 1e4, 5e4, 1e5],
         clf__gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          )
pipe = Pipeline(estimators)
clf = GridSearchCV(pipe, param_grid=param_grid)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
y_true, y_pred = labels_test, pred
print classification_report(y_true, y_pred)
print "\nRecall Score:", recall_score(labels_test, pred)
print "\nPrecision Score:", precision_score(labels_test, pred)

### Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
dectree_score = clf.score(features_test, labels_test)
print '\nAccuracy', dectree_score


### RandomForest
from sklearn.ensemble import RandomForestClassifier
estimators = [
              ('reduce_dim', PCA(n_components=3)),
              ('clf', RandomForestClassifier(random_state = 43))]
param_grid = dict(
         clf__n_estimators = [5,6,7]
          )
pipe = Pipeline(estimators)
clf = GridSearchCV(pipe, param_grid=param_grid)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print clf.best_estimator_
test_classifier(clf, my_dataset, features_list, folds = 1000)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
