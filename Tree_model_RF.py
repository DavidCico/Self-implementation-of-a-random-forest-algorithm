from random import randrange


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# GINI index as cost function to minimize
def gini_index(groups,classes):
    #count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    #Gini indexes weighted sum
    Gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0.0:
            continue
        score = 0.0
        #score group based on score of each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p*p
        #weight group score b yrelative size
        Gini += (1.0-score) * (size / n_instances)
    return Gini

#Split a dataset
def test_split(index,value,dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index]< value:
            left.append(row)
        else:
            right.append(row)
    return left, right

#Best splitpoint for dataset with choice of attributes to avoid redundancy
def get_split_forest(dataset, n_attributes):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    attributes = list()
    while len(attributes) < n_attributes:
        index = randrange(len(dataset[0])-1)
        if index not in attributes:
            attributes.append(index)
    for index in attributes:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            Gini = gini_index(groups, class_values)
            #print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], Gini))
            if Gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], Gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create a function to split a node in function of diffrent parameters
def split_node_forest(node, max_depth, min_size, n_attributes, depth):
    left, right = node['groups']
    #Delete data from node as it is no longer needed
    del(node['groups'])
    #Check whether left node or right node is empty to create a terminal node
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    #Check if we have reached the maximum depth of the tree --> Create terminal node
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    #Process on the both children, by checking if min size is reached first or further split of tree is required
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split_forest(left, n_attributes)
        split_node_forest(node['left'], max_depth, min_size, n_attributes,depth+1)
    #Right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split_forest(right, n_attributes)
        split_node_forest(node['right'], max_depth, min_size, n_attributes,  depth+1)

# Build a decision tree
def build_tree(train, max_depth, n_attributes ,min_size):
    root = get_split_forest(train, n_attributes)
    split_node_forest(root, max_depth, min_size, n_attributes ,1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)



# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)


