# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, f1_score 
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

"""
Define functions to get Euclidean, Manhattan & Chebyshev Distances between two
points in a matrix.
These will be used for outlier identification by the Outlier-SMOTE algorithm.
These will not be used for neighbour identification by either algorithm as the
library used to identify neighbours will allow distance metric to be passed as
an argument.
"""

# Euclidean Distance
def get_euclidean_distance(matrix, point_1, point_2):
    v1 = matrix[point_1]
    v2 = matrix[point_2]
    return np.sqrt(((v1-v2)**2).sum())

# Manhattan Distance
def get_manhattan_distance(matrix, point_1, point_2):
    v1 = matrix[point_1]
    v2 = matrix[point_2]
    return (abs(v1 - v2)).sum()

# Chebyshev Distance
def get_chebyshev_distance(matrix, point_1, point_2):
    v1 = matrix[point_1]
    v2 = matrix[point_2]
    return (abs(v1 - v2)).max()
  
"""
SMOTE class

Arguments at instantiation:

original samples - a numpy array of X attributes (no labels) of minority instances
N - oversampling rate: 100 indicates return as many synthetic instances as there are 
    existing minority instances passed as X. 200 indicates returning twice as many, etc.
k - number of nearest neighbours to find for each instance when oversampling
random_seed - the random seed set before shuffling X
metric - the distance metric to be applied for finding neighbours

oversample method returns:

self.synthetic - a numpy array containing synthetic instances. When N=100, the number of 
    synthetic instances is exactly equal to the number of non-sythetic minority instances 
    passed at construction of the SMOTE object.
    
Notes:

This class does not take class labels as an argument.
This class does not return class labels. As the intended use is to create synthetic 
    minority samples, it should be assumed that a label of "1" can be assigned 
    later to all instances returned by the oversample method.
This class does not perform any type of normalisation or standardisation. This should
    be carried out prior to passing data to the constructor of the class.
"""

class SMOTE:
    def __init__(self, original_samples, N, k, random_seed=1, metric="euclidean"):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.samples = original_samples
        np.random.shuffle(self.samples) # shuffle minority instances
        self.N = N // 100 # 100% converts to 1 here
        self.k = k # number of neighbours required
        self.T = np.shape(original_samples)[0] # number of minority class samples
        self.n_attributes = np.shape(original_samples)[1] # number of attributes
        self.synthetic = np.zeros((self.T * self.N, self.n_attributes)) 
        self.new_index = 0
        self.neighbors = NearestNeighbors(n_neighbors=k, # get neighbours
                                        metric=metric).fit(self.samples)
    # Oversample method
    def oversample(self):
        for i in range(self.T): # loop through each sample in minority class
            sample = self.samples[i]
            nn_array = self.neighbors.kneighbors(sample.reshape(1,-1),
                                           return_distance=False)[0]
            for j in range(self.N): # loop through a number of neighbours determined by N
                chosen_index = random.randint(0, self.k - 1) # choose a neighbour
                delta = self.samples[nn_array[chosen_index]] - sample # find difference
                distance = np.random.rand(delta.shape[0]) # randomised distance in each dimension
                change = np.multiply(delta, distance) # multiply randomised distance by difference
                synthetic_sample = sample + change # add to sample to create synthetic sample
                self.synthetic[self.new_index] = synthetic_sample # place synthetic sample in array
                self.new_index += 1
        return self.synthetic

"""
Outlier-SMOTE class

Arguments at instantiation:

original samples - a numpy array of X attributes (no labels) of minority instances
N - oversampling rate: 100 indicates intention to return as many synthetic instances as 
    there are existing minority instances passed as X. 200 indicates returning twice as many, 
    etc. Note that due to rounding, the exact number of returned instances will vary 
    a little.
k - number of nearest neighbours to find for each instance when oversampling
random_seed - the random seed set before shuffling X
metric - the distance metric to be applied for finding neighbours and also to
    identification of outliers

oversample method returns:

self.synthetic - a numpy array containing synthetic instances. When N=100, the number of 
    synthetic instances is roughly equal to the number of non-sythetic minority instances 
    passed at construction of the SMOTE object.
    
Notes:

This class does not take class labels as an argument.
This class does not return class labels. As the intended use is to create synthetic 
    minority samples, it should be assumed that a label of "1" can be assigned 
    later to all instances returned by the oversample method.
This class does not perform any type of normalisation or standardisation. This should
    be carried out prior to passing data to the constructor of the class.
Passing any metric other than "euclidean", "manhattan" or "chebyshev" will cause failure
    when oversample method is called as self.get_distance will not point to a defined metric.
"""


class Outlier_SMOTE:
    def __init__(self, original_samples, N, k, metric="euclidean", random_seed=1):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.samples = original_samples
        np.random.shuffle(self.samples) # shuffle X
        self.N = N // 100 # 100% converts to 1 here
        self.k = k # number of neighbours
        self.T = np.shape(original_samples)[0] # number of minority class samples
        self.n_attributes = np.shape(original_samples)[1] # number of attributes
        self.synthetic = np.zeros((self.T * self.N, self.n_attributes))
        self.new_index = 0
        self.metric = metric # set distance metric for neighbours
        self.neighbors = NearestNeighbors(n_neighbors=k,
                                      metric=self.metric).fit(self.samples)
        # set the distance metric for outlier identification
        if metric == "euclidean":
            self.get_distance = get_euclidean_distance
        elif metric == "manhattan":
            self.get_distance = get_manhattan_distance
        elif metric == "chebyshev":
            self.get_distance = get_chebyshev_distance
    
    # Oversample method
    def oversample(self):
        # instantiate a distance metric of correct dimensions n x n
        distance_matrix = np.zeros((self.T, self.T))
        # populate the distance matrix
        for i in range(self.T):
            for j in range(self.T):
                if j > i:
                    distance = self.get_distance(self.samples, i, j)
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
        # Use distance matrix to form columnwise sum matrix
        column_wise_sum_matrix = distance_matrix.sum(axis=0)
        # Obtain normalised version of columnwise sum matrix
        normalised_matrix = column_wise_sum_matrix / sum(column_wise_sum_matrix)
        # Construct and round oversampling matrix from normalised columnwise sum matrix
        oversampling_matrix = np.round(self.N * self.T * normalised_matrix).astype(int)
        # Find number of samples required
        samples_required = oversampling_matrix.sum()
        # re-set size of self.synthetic, may be slightly different due to rounding
        self.synthetic = np.zeros((samples_required, self.n_attributes))
        # oversampling list to indicate how many times each instance will be oversampled
        oversampling_list = list(oversampling_matrix)

        for index, element in enumerate(oversampling_list):
            if element == 0: # when no oversampling of this element required
                continue
            sample = self.samples[index] # select sample
            # get sample's neighbours
            nn_array = self.neighbors.kneighbors(sample.reshape(1,-1),
                                           return_distance=False)[0]
            for i in range(element): # oversample the instance as per oversampling matrix
                chosen_index = random.randint(0, self.k - 1) # choose a neighbour
                delta = self.samples[nn_array[chosen_index]] - sample # difference from sample-neighbour
                distance = np.random.rand(delta.shape[0]) # randomised distance in each dimension 
                change = np.multiply(delta, distance) # multiply difference by random distance
                synthetic_sample = sample + change # create synthetic sample
                self.synthetic[self.new_index] = synthetic_sample # place synthetic sample to array
                self.new_index += 1
        return self.synthetic
    
"""
experiment function: runs one experiment

Arguments: 

X - A numpy array containing all instances without labels
y - A numpy array containing corresponding labels
classifier - default to Random Forest with 100 Decision Trees, non-defaults not to be used
oversampler - defaults to False, accepts "Outlier_SMOTE" or "SMOTE" as arguments. Causes a 
    string describing error to be returned by experiment if a different argument is given.
n_splits - number of cross-validation folds, default = 10
random_state - the random seed to be set during this experiment to allow pseudo-random shuffling 
    etc with replicability as required.
metric - The distance metric to be used. Defaults to "euclidean", also accepts "chebyshev" and 
    "manhattan". If a different argument is given, the experiment will return an string describing
    the error.
N - oversampling rate, defaults to 100
k - number of neighbours to be found for oversampling purposes, defaults to 5

Returns:

Performance metrics for each cross-validation fold
"""


def experiment(X, y, classifier="rf100", over_sampler=False, 
               n_splits=10, random_state=1, metric="euclidean", N=100, k=5):
    # set classifier
    if classifier == "rf100": # classifier as stated in report
        clf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif classifier == "rf500": # classifier for purposes of initial exploration
        clf = RandomForestRegressor(max_depth=16, max_features=19, 
                                   min_samples_leaf=2, min_samples_split=8, 
                                   n_estimators=500)
    else:
        return "Classifier input error"
  
    # instantiate StraitfiedKfolds and output list
    metrics_list = []
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                            random_state=random_state)

    # loop through 10 cv folds
    for train_index, test_index in skfolds.split(X, y):
        # metrics dict and classifier
        metrics = {}
        clone_clf = clone(clf)
        # train and test sets for this fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        # scale dataset
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.transform(X_train)
        scaler.transform(X_test)
        # set over-sampler if necessary
        if over_sampler != False:
            pos_indices = np.where(y_train == 1)
            X_train_positive = np.take(X_train, pos_indices[0], axis=0)
            if over_sampler == "SMOTE":
                smote = SMOTE(original_samples=X_train_positive, N=N, k=k, 
                                random_seed=random_state, metric=metric)
            elif over_sampler == "Outlier_SMOTE":
                smote = Outlier_SMOTE(original_samples=X_train_positive, N=N, k=k, 
                                random_seed=random_state, metric=metric)
            else: # Not left as false and neither desired argument
                return "SMOTE input error"
            # oversample if necessary
            X_synthetic = smote.oversample()
            n_synthetic = np.shape(X_synthetic)[0]
            y_synthetic = np.ones(n_synthetic)
            X_train = np.concatenate((X_train, X_synthetic))
            y_train = np.concatenate((y_train, y_synthetic))
            # oversampling complete
        clone_clf.fit(X_train, y_train)
        y_predicted_probabilities = clone_clf.predict(X_test)
        y_predicted_classes = np.where(y_predicted_probabilities > 0.5, 1, 0)
        conf_matrix = confusion_matrix(y_test, y_predicted_classes)
        metrics["confusion_matrix"] = conf_matrix
        metrics["ROCAUC"] = roc_auc_score(y_test, y_predicted_probabilities)
        metrics["f1"] = f1_score(y_test, y_predicted_classes)
        metrics["precision"] = precision_score(y_test, y_predicted_classes)
        metrics["recall"] = recall_score(y_test, y_predicted_classes)
        metrics_list.append(metrics)
        # end of this fold
    # end of looping through folds
    return metrics_list

"""
ten_experiments function: performs ten experiments as per function above

Arguments:

X - A numpy array containing all instances without labels
y - A numpy array containing corresponding labels
classifier - default to Random Forest with 100 Decision Trees, non-defaults not to be used
oversampler - defaults to False, accepts "Outlier_SMOTE" or "SMOTE" as arguments. The experiment
    fucntion causes a string describing error to be returned by experiment if a different 
    argument is given.
n_splits - number of cross-validation folds for each of the ten experiments, default = 10
random_state - the random seed to be passed for the first of 10 experiment to allow pseudo-random            
    shuffling etc with replicability as required. Each subsequent experiment will be passed a random
    state one higher than the previous.
metric - The distance metric to be used. Defaults to "euclidean", also accepts "chebyshev" and 
    "manhattan". If a different argument is given, the experiment function will return an string describing the error.
N - oversampling rate, defaults to 100
k - number of neighbours to be found for oversampling purposes, defaults to 5

Returns: a dictionary of metrics output by each experiment.

"""

def ten_experiments(X, y, classifier="rf100", over_sampler=False, 
               n_splits=10, random_state=1, metric="euclidean", N=100, k=5):
    output = {}
    for experiment_number in range(random_state, random_state + 10):
        metrics = experiment(X, y, classifier=classifier, over_sampler=over_sampler, 
                    n_splits=n_splits, random_state=experiment_number, metric=metric, N=N, k=k)
        output[experiment_number] = metrics
    return output
    
    
"""
get_DataFrame_from_results function: converts the output of the ten_experiments function to a
Pandas DataFrame

Arguments: res_dict - a results dictionary as returned by ten_experiments function

returns: a Pandas DataFrame summarising experimental results
"""
    
def get_DataFrame_from_results(res_dict):
    col_names = ['TN', 'FN', 'TP', 'FP', 'ROCAUC', 'f1', 'precision', 'recall']
    data = pd.DataFrame(columns = col_names)
    for exp_number in range(1, 11):
        entry = res_dict[exp_number]
        for result in entry:
            working_list = []
            CM = result['confusion_matrix']
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            working_list.append(TN)
            working_list.append(FN)
            working_list.append(TP)
            working_list.append(FP)
            working_list.append(result['ROCAUC'])
            working_list.append(result['f1'])
            working_list.append(result['precision'])
            working_list.append(result['recall'])
            working_series = pd.Series(working_list, index=col_names)
            data = data.append(working_series, ignore_index=True)
    return data

"""
experimental_datasets function: mimics experiment function without training a classifier.
    De-normalises the training data (real and synthetic) for each fold of an experiment.
    Saves this data to disc.
    
Arguments:

X - instances without labels as numpy array
y - labels of corresponding instances as numpy array
over-sampler - defaults to False, allows SMOTE or Outlier_SMOTE to be set
exp_reference - as this will be called by a ten experimental datasets function, this reference
    exists to keep track of which numbered experiment the data refers to, defaults to 1.
column_names - the attribute names for each label in X
dataset_name - the name of the dataset being oversampled. Default is "Abalone".
n_splits - number of cross-validation folds, defaults to 10.
random_state - random seed to be set to ensure same datasets produced as in experiments. Default is 1.
metric - the Distance Metric to be employed.
N - oversampling rate
k - number of nearest neighbours

Actions:

Saves every produced training set to disc with naming format identifying dataset, over-sampler, 
    oversampling rate, experiment number, cross-validation fold number

"""
  
def experimental_datasets(X, y, over_sampler=False, exp_reference=1, 
                column_names=[], dataset_name="Abalone",
                n_splits=10, random_state=1, metric="euclidean", N=100, k=5):
  
    # instantiate StraitfiedKfolds and output list
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                            random_state=random_state)
    # keep track of fold number
    fold_ref = 0
    # loop through 10 cv folds
    for train_index, test_index in skfolds.split(X, y):
        fold_ref += 1 # first fold is fold 1, second is fold 2 etc
        # train and test sets for this fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        # scale dataset
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.transform(X_train)
        scaler.transform(X_test)
        # set over-sampler if necessary
        if over_sampler != False:
            pos_indices = np.where(y_train == 1)
            X_train_positive = np.take(X_train, pos_indices[0], axis=0)
            if over_sampler == "SMOTE":
                smote = SMOTE(original_samples=X_train_positive, N=N, k=k, 
                                random_seed=random_state, metric=metric)
            elif over_sampler == "Outlier_SMOTE":
                smote = Outlier_SMOTE(original_samples=X_train_positive, N=N, k=k, 
                                random_seed=random_state, metric=metric)
            else: # Not left as false and neither desired argument
                print("SMOTE input error")
                return "SMOTE input error"
            # oversample 
            X_synthetic = smote.oversample()
            n_synthetic = np.shape(X_synthetic)[0]
            y_synthetic = np.ones(n_synthetic)
            X_train = np.concatenate((X_train, X_synthetic))
            y_train = np.concatenate((y_train, y_synthetic))
            # oversampling complete
            # inverse scale transform X_train
            X_train = scaler.inverse_transform(X_train)
            # DataFrame from oversampled X_train
            data = pd.DataFrame(X_train, columns=column_names)
            data['y'] = y_train
            # Save csv file from constructed DataFrame of original and synthetic data
            name = dataset_name + "-" + over_sampler + "-" + metric + "-" + str(N)  
            name += "-exp-" + str(exp_reference) + "-fold-" + str(fold_ref) + ".csv"
            data.to_csv(name)
            print(f"saved {name} to drive")
            # end of this fold
            
"""
ten_experimental_datasets function: mimics ten_experiments function without training a classifier.
    Calls function which de-normalises the training data (real and synthetic) for each 
    fold of an experiment and saves this data to disc.
    
Arguments:

X - instances without labels as numpy array
y - labels of corresponding instances as numpy array
over-sampler - defaults to False, allows SMOTE or Outlier_SMOTE to be set
exp_reference - as this will be called by a ten experimental datasets function, this reference
    exists to keep track of which numbered experiment the data refers to, defaults to 1.
column_names - the attribute names for each label in X
dataset_name - the name of the dataset being oversampled. Default is "Abalone".
n_splits - number of cross-validation folds, defaults to 10.
random_state - random seed to be set to ensure same datasets produced as in experiments. Default is 1.
    First experiment would be called with 1 as random_state, second with 2 etc.
metric - the Distance Metric to be employed.
N - oversampling rate
k - number of nearest neighbours

Actions:

Calls experimental_datasets function ten times. Using random states to mimic the ten_experiments
    function.
"""

def ten_experimental_datasets(X, y, over_sampler=False, column_names = [],
                dataset_name="Abalone", n_splits=10, random_state=1, 
                metric="euclidean", N=100, k=5):
    for experiment_number in range(random_state, random_state + 10):
        experimental_datasets(X, y, over_sampler=over_sampler, 
                column_names=column_names, dataset_name=dataset_name,
                exp_reference=experiment_number,
                n_splits=n_splits, random_state=experiment_number, 
                metric=metric, N=N, k=k)