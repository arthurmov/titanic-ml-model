# Arthur Movsesyan
# CISC 3440: Machine Learning
# P1
#
# We worked on Kaggle competition Dataset (https://www.kaggle.com/competitions/titanic/data?select=train.csv) for this project. We used Decistion Tree Classifer to predict if a passenger on Titanic ship would survive or not.
# **Data Features:** Each line in the file has the following Features
# 
# 
# | Variable  | Definition                                   | Key                                      |
# |-----------|----------------------------------------------|------------------------------------------|
# | survival  | Survival -Label                                     | 0 = No, 1 = Yes                          |
# | pclass    | Ticket class                                 | 1 = 1st, 2 = 2nd, 3 = 3rd                |
# | sex       | Sex                                          |                                          |
# | Age       | Age in years                                 |                                          |
# | sibsp     | # of siblings / spouses aboard the Titanic   |                                          |
# | parch     | # of parents / children aboard the Titanic   |                                          |
# | fare      | Passenger fare                               |                                          |
# | embarked  | Port of Embarkation                          | C = Cherbourg, Q = Queenstown, S = Southampton |
# 
# **First Row**
# 
# | Survived | Pclass | Sex  | Age | SibSp | Parch | Fare | Embarked |
# |----------|--------|------|-----|-------|-------|------|----------|
# | 0        | 3      | male | 22  | 1     | 0     | 7.25 | S        |
# 
# 
# The function below reads the data and qunatizes them using the following criteria:
# 
#     - Pclass: remains as an integer.
#     - Sex: 'male' is mapped to 0, 'female' to 1.
#     - Age: divided by 10 and truncated to an integer.
#     - SibSp: remains as an integer.
#     - Parch: remains as an integer.
#     - Fare: remains as a floating-point number.
#     - Embarked: 'S' is mapped to 0, 'C' to 1, 'Q' to 2, and unknown values to

# Global Constants for column indices
PCLASS = 0
SEX = 1
AGE = 2
SIBSP = 3
PARCH = 4
FARE = 5
EMBARKED = 6
attribute_list = ["PCLASS", "SEX", "AGE", "SIBSP", "PARCH", "FARE", "EMBARKED"]


import numpy as np

def get_data_and_labels(filename, contains_label=True):
    '''
    This function reads data from a CSV file, processes it, and returns two numpy arrays:
    one with quantized features and another with labels (if labels are present).

    ARGS:

    filename : str
        The path to the CSV file containing the data.

    contains_label : bool, optional
        A flag indicating whether the input file contains labels (default is True).

    Returns:
    quantized_features : list of list
        A list of lists where each inner list contains the quantized features for each data entry.

    labels : list
        A list containing labels for each data entry if `contains_label` is True.

    Note:
    The function assumes that the input CSV file has the following columns in order:
    Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

    The quantization process:
    - Pclass: remains as an integer.
    - Sex: 'male' is mapped to 0, 'female' to 1.
    - Age: divided by 10 and truncated to an integer.
    - SibSp: remains as an integer.
    - Parch: remains as an integer.
    - Fare: remains as a floating-point number.
    - Embarked: 'S' is mapped to 0, 'C' to 1, 'Q' to 2, and unknown values to -1.
    '''


    # Open the file and read all lines except the header
    file = open(filename, 'r')
    data = file.readlines()[1:]  # Skip the header

    # Extract labels and features if the file contains labels
    if contains_label:
        labels = [int(x.strip('\n').split(',')[0]) for x in data]
        features = [x.strip('\n').split(',')[1:] for x in data]
    else:
        features = [x.strip('\n').split(',') for x in data]

    quantized_features = []

    # Process each feature and quantize them
    for feature in features:
        quantized_feature = []
        for column in range(len(feature)):
            if column == PCLASS:
                quantized_feature.append(int(feature[column]))  # Pclass: as is
            elif column == SEX:
                if feature[column] == 'male':
                    quantized_feature.append(0)  # Sex: 'male' -> 0
                else:
                    quantized_feature.append(1)  # Sex: 'female' -> 1
            elif column == AGE:
                try:
                    quantized_feature.append(int(float(feature[column]) // 10))  # Age: decade-wise quantization
                except:
                    quantized_feature.append(-1)  # Missing age: represented as -1
            elif column == SIBSP:
                quantized_feature.append(int(feature[column]))  # SibSp: as is
            elif column == PARCH:
                quantized_feature.append(int(feature[column]))  # Parch: as is
            elif column == FARE:
                quantized_feature.append(int(float(feature[column]) // 30))  # Fare: Quantized as multiples of 30
            else:
                # Embarked: map 'S' -> 0, 'C' -> 1, 'Q' -> 2, unknown -> -1
                if feature[column] == 'S':
                    quantized_feature.append(0)
                elif feature[column] == 'C':
                    quantized_feature.append(1)
                elif feature[column] == 'Q':
                    quantized_feature.append(2)
                else:
                    quantized_feature.append(-1)
        quantized_features.append(quantized_feature)

    return quantized_features, labels if contains_label else None


# **Computing Entropy**
# The function below has the skeleton to fill in the code to compute entropy. Entropy is based on the porbability distribiution of different values for the labels
# 
# The entropy \( $H(L)$ \) is defined as:
# 
# $H(L) = -\sum_{i=1}^{c} p_i \log_2(p_i)$
# 
# where:
# - \( $p_i$ \) is the probability of class \( $i$ \),
# - \( $c$ \) is the total number of classes.
# - L is our label probability distribution
# 

import math as m
features, labels = get_data_and_labels('train.csv')

def compute_entropy(labels):
  """
    Computes the entropy of a set of labels.

    Args:
        labels (list or array-like): The list of labels for which entropy is to be computed.

    Returns:
        float: The entropy of the label set.
  """

  entropy = 0

  probability_survive = len([x for x in labels if x == 0]) /(len(labels))
  probability_not_survive = len([x for x in labels if x == 1]) /(len(labels))

  if probability_survive > 0:
        entropy -= probability_survive * m.log2(probability_survive)
  if probability_not_survive > 0:
        entropy -= probability_not_survive * m.log2(probability_not_survive)
  return entropy

compute_entropy(labels)


# **Splitting the dataset by feature value**

def split_by_feature_value(data, labels, feature_name):
    """
    Splits the dataset into subsets based on unique values of the specified feature.

    ARGS:

    data : np.ndarray
        A 2D numpy array where each row represents a data instance and each column
        represents a feature.
    labels : list or np.ndarray
        A 1D array or list containing the labels corresponding to the data instances.
    feature_name : int
        The index of the feature (column) by which to split the data.

    Returns:

    spilt_labels : dict
        A dictionary where the keys are the unique values of the specified feature, and
        the values are numpy arrays of labels corresponding to each unique feature value.
    """

    # Initialize an empty dictionary to hold the split labels
    spilt_labels = {}
    split_data={}

    # Get the unique values of the specified feature
    feature_values = set(data[:, feature_name].tolist())

    # Iterate over each unique feature value
    for feature_value in feature_values:
        # Get the labels corresponding to the current feature value
        spilt_labels[feature_value] = np.array(labels)[np.argwhere(data[:, feature_name] == feature_value).reshape(-1)].tolist()
        split_data[feature_value]=data[np.argwhere(data[:, feature_name] == feature_value).reshape(-1)]

    return spilt_labels,split_data


# **Computting Information Gain**
# 
# $Information Gain(L,A)=Entropy(L)- \sum_v\frac{L_v}{L}Entropy(L_v)$
# Where:
#   - Entropy (L) - should use the function compute_entropy()
#   - L - The labels without split based on attribute values
#   - A - is the attribute
#   - v - represents the different values that the attribute can take
#   - $L_v$ - reprensets the labels correspdoning to the attribute A with value v


import numpy as np
features, labels = get_data_and_labels('train.csv')
def compute_information_gain(data, labels, feature_name):
  """
    Computes the information gain of a given feature in the dataset.

    Args:
        data (list or array-like): The dataset containing the features.
        labels (list or array-like): The list of labels corresponding to each data point.
        feature_name (str): The name of the feature or attribute for which information gain is computed.

    Returns:
        float: The information gain achieved by splitting the dataset using the given feature.
    """
  
  original_entropy = compute_entropy(labels)

  split_labels, split_data = split_by_feature_value(np.array(data), labels, feature_name)

  weighted_entropy = 0
  total_count = len(labels)

  for feature_value, subset_labels in split_labels.items():
    
    subset_proportion = len(subset_labels) / total_count
    
    subset_entropy = compute_entropy(subset_labels)
    
    weighted_entropy += subset_proportion * subset_entropy

  information_gain = original_entropy - weighted_entropy

  
  return information_gain


# **Creating Training and Validation Data splits**
# 
# The function below will take the dataset and labels and create random splits based on given percentage.


def create_training_validation_split(data,labels,split_percentage):
  """
    Splits a dataset into training and validation sets based on a given split percentage.

    ARGS:
    ----------
    data : numpy.ndarray
       The data to be split into train and validation

    labels : list
        The corresponding labels for the data

    split_percentage : float
        A float between 0 and 1 that specifies the proportion of the dataset to allocate to the training set.
        For example, 0.8 would allocate 80% of the data to the training set and the remaining 20% to the validation set.

    Returns:
    -------
    tuple : (train_features, train_labels, validation_features, validation_labels)
        - train_features : numpy.ndarray
            The feature data for the training set.
        - train_labels : numpy.ndarray
            The corresponding labels for the training set.
        - validation_features : numpy.ndarray
            The feature data for the validation set.
        - validation_labels : numpy.ndarray
            The corresponding labels for the validation set.
  """
  import random
  import numpy as np
  train_features=[]
  train_labels=[]
  validation_features=[]
  validation_labels=[]
  total_sample_count=len(data)
  train_sample_count=int(split_percentage*total_sample_count)
  validation_sample_count=total_sample_count-train_sample_count
  train_indices=random.sample(range(total_sample_count),train_sample_count)
  validation_indices=random.sample(list(set(range(total_sample_count))-set(train_indices)),validation_sample_count)
  for i in range(total_sample_count):
    if i in train_indices:
      train_features.append(data[i])
      train_labels.append(labels[i])
    if i in validation_indices:
      validation_features.append(data[i])
      validation_labels.append(labels[i])
  return np.array(train_features),np.array(train_labels),np.array(validation_features),np.array(validation_labels)


# **Building the Decision Tree**
# 
# Using Sklearn Decision Tree library. We perform hyper-parameter tuning using the provided data.

import pickle
from sklearn import tree
import graphviz
from IPython.display import Image
quantized_features, labels=get_data_and_labels('train.csv')
train_features,train_labels,validation_features,validation_labels=create_training_validation_split(quantized_features,labels,0.9)

classifier = tree.DecisionTreeClassifier(criterion='entropy')

classifier.fit(train_features, train_labels)

dot_data = tree.export_graphviz(classifier, out_file=None)
graph = graphviz.Source(dot_data)
Image(graph.pipe(format='png'))