#Import and Conversion, Normalization of Data
from Open_Conversion_Data import load_csv
from Open_Conversion_Data import str_column_to_float
from Open_Conversion_Data import str_column_to_int

#Algorithm evaluation with different steps 
from Algorithm_test_harness import evaluate_algorithm_cv

#
from Performance_assessment import getAccuracy

#Import math/random functions
from math import sqrt
from random import seed

#Import random forest tree model
from Tree_model_RF import random_forest

def main():
    # Test the random forest algorithm on sonar dataset
    seed(2)
    # load and prepare data
    filename = 'sonar-all-data.csv'
    dataset = load_csv(filename)
    # convert string attributes to integers
    for i in range(0, len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
    n_folds = 5
    max_depth = 10
    min_size = 1.0
    sample_size = 1.0
    n_features = int(sqrt(len(dataset[0])-1))
    for n_trees in [1, 5, 10]:
        scores = evaluate_algorithm_cv(dataset, random_forest, n_folds, getAccuracy,max_depth, min_size, sample_size, n_trees, n_features)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

main()