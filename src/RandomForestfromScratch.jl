"""
    Steps to build random forests:

    1 - Create bootstrap data: same size as the original dataset - randomly select the data (rows) from the dataset (with replacement)
    
    
    2- Create a decision tree with the bootstrap data using only a random subset of the features (columns)
                                                                                            at each split in the same tree until a decision is made
    
    3- Repeat step 1 and 2
    
    
    4- Predict function

"""

# RandomForest.jl

include("ClassificationTree.jl")

using StatsBase: mode, sample, countmap
using Random
using Statistics

include("ClassificationTree.jl")

"""
Represents a RandomForest for classification.

`n_trees` is the number of trees in the forest.

`max_depth` controls the maximum depth of each tree. If -1, the depth is not limited.

`min_samples_split` controls when a node in the decision tree should be split.

`max_features` controls the number of features to consider when looking for the best split.

`trees` is a vector containing the individual ClassificationTree instances.

`data` contains the datapoints of the RandomForest.

`labels` contains the respective labels of the datapoints.
"""
mutable struct RandomForest{T, L}
    n_trees::Int  # the number of trees in the forest
    max_depth::Int # the maximum depth of each tree. If -1, the depth is not limited
    min_samples_split::Int # when a node in the decision tree should be split
    max_features::Int # the number of features to consider when looking for the best split

    trees::Vector{ClassificationTree{T, L}} # has all the individual ClassificationTree instances

    data::Matrix{T}
    labels::Vector{L}
    
    RandomForest(n_trees::Int, max_depth::Int, min_samples_split::Int, max_features::Int, data::Matrix{T}, labels::Vector{L}) where {T, L} = 
        size(data, 1) == length(labels) ? new{T, L}(n_trees, max_depth, min_samples_split, max_features, ClassificationTree{T, L}[], data, labels) : throw(ArgumentError("The number of rows in data must match the number of elements in labels"))
end

function fit_forest(rf::RandomForest)
    for i in 1:rf.n_trees
        indices = sample(1:size(rf.data, 1), size(rf.data, 1), replace=true)
        bootstrap_data = rf.data[indices, :]
        bootstrap_labels = rf.labels[indices]
        tree = ClassificationTree(rf.max_depth, rf.min_samples_split, bootstrap_data, bootstrap_labels)
        fit_rf_tree(tree, rf.max_features)
        push!(rf.trees, tree)
    end
end


"""
    predict_forest(rf::RandomForest, data::Matrix{T})

Returns the prediction of the RandomForest for a list of datapoints.

`data` contains the datapoints to predict.
"""
function predict_forest(rf::RandomForest, data::Matrix{T}) where {T}
    tree_predictions = [predict(tree, data) for tree in rf.trees]
    predictions = []

    for i in 1:size(data, 1)
        sample_predictions = [tree_predictions[j][i] for j in 1:rf.n_trees]
        push!(predictions, mode(sample_predictions))
    end
    
    return predictions
end

function find_best_split_rf(data::Matrix{T}, labels::Vector{L}, max_features::Int) where {T, L}
    best_gini = Inf
    best_feature_index = -1
    best_feature_value = -1
    num_features = size(data, 2)
    
    features = sample(1:num_features, max_features, replace=false)
    
    for feature_index in features
        unique_values = unique(data[:, feature_index])
        for value in unique_values
            left_labels, right_labels = split_node(data, labels, feature_index, value)
            gini = weighted_gini(left_labels, right_labels)
            if gini < best_gini
                best_gini = gini
                best_feature_index = feature_index
                best_feature_value = value
            end
        end
    end
    return (best_feature_index, best_feature_value)
end

function build_rf_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, max_features, depth::Int=0) where {T, L}
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth)
        return Leaf{L}(labels)
    end

    if allequal(labels)
        return Leaf{L}(labels)
    end
    
    feature_index, split_value = find_best_split_rf(data, labels, max_features)
    
    if isa(split_value, Number)
        left_mask = data[:, feature_index] .< split_value
        right_mask = data[:, feature_index] .>= split_value
    else
        left_mask = data[:, feature_index] .!= split_value
        right_mask = data[:, feature_index] .== split_value
    end

    left_data, right_data = data[left_mask, :], data[right_mask, :]
    left_labels, right_labels = labels[left_mask], labels[right_mask]

    node = Node(data, labels)
    node.feature_index = feature_index
    node.split_value = split_value

    node.left = build_rf_tree(left_data, left_labels, max_depth, min_samples_split, max_features, depth + 1)
    node.right = build_rf_tree(right_data, right_labels, max_depth, min_samples_split, max_features, depth + 1)

    return node
end

function fit_rf_tree(tree::ClassificationTree, max_features::Int=size(tree.data, 2))
    tree.root = build_rf_tree(tree.data, tree.labels, tree.max_depth, tree.min_samples_split, max_features)
end

# Create synthetic data
X = [
    1 0 5;
    2 3 1;
    5 2 3;
    0 1 4;
    3 5 0;
    4 1 2;
    1 3 3;
    2 4 0;
    5 1 1;
    0 2 5
]

y = [10, 15, 20, 8, 25, 18, 17, 13, 22, 12]

# Create and build a RandomForest
rf = RandomForest(10, 5, 2, 2, X, y)
fit_forest(rf)

# Predict with the RandomForest
predictions = predict_forest(rf, X)
println(predictions)

data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
rf = RandomForest(10, -1, 1, 2, data, labels)
fit_forest(rf)
test_data = ["dog" 38.0; "human" 38.0]
prediction = predict_forest(rf, test_data)
println(prediction)

# Helper function from GiniImpurity class - should be deleted later
function gini_impurity(labels::Vector{L}) where {L}
    label_counts_dict = countmap(labels)
    total = length(labels)
    if total == 0
        return 0.0
    end
    label_probabilities = values(label_counts_dict) ./ total
    gini = 1 - sum(label_probabilities.^2)
    return gini
end

function weighted_gini(left_dataset::Vector{L}, right_dataset::Vector{L}) where {L}
    number_of_left_rows = length(left_dataset)
    number_of_right_rows = length(right_dataset)
    total = number_of_left_rows + number_of_right_rows
    gini_split = (number_of_left_rows / total) * gini_impurity(left_dataset) + (number_of_right_rows / total) * gini_impurity(right_dataset)
    return gini_split
end

function split_node(data::Matrix{T}, labels::Vector{L}, index, value) where {T, L}
    x_index = data[:, index]
    if eltype(x_index) <: Number
        mask = x_index .>= value
    else
        mask = x_index .== value
    end
    left = labels[.!mask]
    right = labels[mask]
    return left, right
end
