from random import randrange

# Split a dataset into a train and test set
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# Split a dataset into $k$ folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Evaluate an algorithm using a train/test split several times
def evaluate_algorithm_tt_split(dataset, algorithm, split, n_splits, performance_assessment,*args):
    scores = list()
    for _ in range(n_splits):
        train, test = train_test_split(dataset, split)
        test_set = list()
        for row in test:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train, test_set, *args)
        actual = [row[-1] for row in test]
        performance = performance_assessment(actual, predicted)
        scores.append(performance)
    return scores

# Evaluate an algorithm using a cross-validation split
def evaluate_algorithm_cv(dataset, algorithm, n_folds, performance_assessment, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        performance = performance_assessment(actual, predicted)
        scores.append(performance)
    return scores