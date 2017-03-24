#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Using a RandomForestClassifier to estimate feature importance
from sklearn.ensemble import RandomForestClassifier

data = featureFormat(data_dict, features_list)
#Keys with all features as zero, which were removed from the dataset by the method above
invalid_keys = ['CLINE KENNETH W', 'PIRO JIM', 'LOCKHART EUGENE E', 'HAYSLETT RODERICK J']

poi, features = targetFeatureSplit( data )

# Using a RandomForestClassifier to estimate the features that are more important
model = RandomForestClassifier(random_state = 1)
model.fit(features, poi)
#Selecting features with importance higher than 8%
#select "Features & Importances that are higher than 8%:"
i = 1
new_features_list = ['poi']
for importance in model.feature_importances_:
    if importance >= 0.08:
        #print features_list[i]
        new_features_list.append(features_list[i])
    i = i + 1
# Most important features selected are: total_payments, exercised_stock_options, bonus, deferred_income, expenses
# and from_this_person_to_poi

features_list = new_features_list

# Adding 'from_this_person_to_poi' and 'from_poi_to_this_person' to features_list to be used later
features_list.append('from_this_person_to_poi')
features_list.append('from_poi_to_this_person')

data = featureFormat(data_dict, features_list )
poi, features = targetFeatureSplit( data )
total_payments = []
exercised_stock_options = []
bonus = []
deferred_income = []
from_this_to_poi = []
from_poi_to_this = []

for f1, f2, f3, f4, f5, f6 in features:
    total_payments.append(f1)
    exercised_stock_options.append(f2)
    bonus.append(f3)
    deferred_income.append(f4)
    from_this_to_poi.append(f5)
    from_poi_to_this.append(f6)

### Task 2: Remove outliers
import matplotlib.pyplot as plt

from sklearn import linear_model
# Create linear regression object
reg = linear_model.LinearRegression()
# Train the model using the training sets
import numpy as np
reg.fit(np.transpose(np.matrix(total_payments)), np.transpose(np.matrix(exercised_stock_options)))

pred = reg.predict(np.transpose(np.matrix(total_payments)))

# Calculating the error
pred_error = abs(pred - np.transpose(np.matrix(total_payments)))
pred_error = pred_error.tolist()
## Remove 7% of the records with maximum errors
error_index_array = []
for i in range(int(len(total_payments)*0.07)):
    index = pred_error.index(max(pred_error))
    error_index_array.append(index)
    del pred_error[index]
## Remove 2 outliers:
## THE TRAVEL AGENCY IN THE PARK
## TOTAL
error_index_array.append(101)
error_index_array.append(98)
invalid_keys.append('THE TRAVEL AGENCY IN THE PARK')
invalid_keys.append('TOTAL')

#print error_index_array
for i in error_index_array:
    del exercised_stock_options[i]
    del total_payments[i]
    del bonus[i]
    del deferred_income[i]
    del poi[i]
    del from_this_to_poi[i]
    del from_poi_to_this[i]

index = 0
for key in data_dict.keys():
    if key in invalid_keys or index in error_index_array:
        del data_dict[key]
    index = index + 1

data = []

for i in range(len(poi)):
    data.append(np.asarray([total_payments[i],exercised_stock_options[i],bonus[i],deferred_income[i]]))

### Task 3: Create new feature(s)

# Will create a new feature based on communication with a POI
# using existing features: 'from_this_person_to_poi' and 'from_poi_to_this_person'

data_with_new_feature = []

# The new feature indicates if there was communication with POI
for i in range(len(poi)):
    communication_with_poi = (from_this_to_poi[i] or from_poi_to_this[i])
    data_with_new_feature.append(np.asarray([total_payments[i],exercised_stock_options[i],
                                             bonus[i],deferred_income[i],communication_with_poi]))

# Use a PCA to create three new features
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def pca_results(features_list, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    Usage: `pca_results(features_list[1:], pca_object)`
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = features_list)
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    plt.figure(figsize=(16,8))
    ax = plt.subplot2grid((10,8), (0, 0), colspan=7, rowspan=8)

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Exp. Var.\n%.4f"%(ev))

    plt.show()
    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

# PCA without scaled data
pca = PCA(n_components=3, svd_solver='auto')
new_data = pca.fit_transform(data, poi)

pca_results(['total_payments','exercised_stock_options','bonus','deferred_income'],pca)

# Create PCA with scaled data
pca = PCA(n_components=4, svd_solver='auto')
new_data_scaled = pca.fit_transform(scale(data), poi)

##Add PCA features to data dictionary
i = 0
for key in data_dict.keys():
    data_dict[key]['PCA_1'] = new_data[i][0]
    data_dict[key]['PCA_2'] = new_data[i][1]
    data_dict[key]['PCA_3'] = new_data[i][2]
    i += 1

### Store to my_dataset for easy export below.

my_dataset = data_dict

# Method to test the algorithm - A Stratified Shuffle Split Cross Validation is used
def test_classifier(clf, features, labels, add_estimators = 10):
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.4, random_state=42)
    sss.get_n_splits(features, labels)

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_idx, test_idx in sss.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.n_estimators += add_estimators
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)

        print "Accuracy:"
        print accuracy
        print "Precision:"
        print precision
        print "Recall:"
        print recall
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# A random forest classifier was chosen
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start='true')
print "Test 1 - Random Forest Classifier without PCA"
test_classifier(clf,data,poi)

clf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start='true')
print "Test 2 - Random Forest Classifier without PCA + new feature"
test_classifier(clf,data_with_new_feature,poi)

clf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start='true')
print "Test 3 - Random Forest Classifier with PCA using Scaled Data"
test_classifier(clf,new_data_scaled,poi)

clf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start='true')
print "Test 4 - Random Forest Classifier with PCA without Scaled Data"
test_classifier(clf,new_data,poi)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Using Grid Search to estimate the best n_estimators parameter
from sklearn.model_selection import GridSearchCV
parameters = {'min_samples_split': range(2,10), 'min_samples_leaf':range(1,10)}

from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.4, random_state=42)

grid_search = GridSearchCV(clf, param_grid=parameters, cv = cv, scoring='recall')
#grid_search.fit(new_data, poi)
#print(grid_search.best_estimator_)

#Result from Grid Search:
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=3, min_weight_fraction_leaf=0.0,
#            n_estimators=110, n_jobs=1, oob_score=False, random_state=42,
#            verbose=0, warm_start='true')

clf = RandomForestClassifier(n_estimators=200, random_state=42,
                             warm_start='true', min_samples_split=2, min_samples_leaf=1)
print "Test 5 - Final Random Forest Classifier"
test_classifier(clf,new_data,poi)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
features_list = ['poi', 'PCA_1', 'PCA_2', 'PCA_3']
dump_classifier_and_data(clf, my_dataset, features_list)