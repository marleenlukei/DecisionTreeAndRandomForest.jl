using StatsBase: mode, sample, countmap
using Random
using Statistics

include("ClassificationTree.jl")

"""
    struct RandomForest{T, L}

Represents a RandomForest for classification and regression.

# Fields
- `n_trees::Int`: The number of trees in the forest.
- `max_depth::Int`: The maximum depth of each tree. If -1, the depth is not limited.
- `min_samples_split::Int`: Controls when a node in the decision tree should be split.
- `max_features::Int`: The number of features to consider when looking for the best split.
- `trees::Vector{ClassificationTree{T, L}}`: A vector containing the individual ClassificationTree instances.
- `data::Matrix{T}`: Contains the datapoints of the RandomForest.
- `labels::Vector{L}`: Contains the respective labels of the datapoints.
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

"""
    fit_forest(rf::RandomForest)

Fits the RandomForest instance by training individual trees on bootstrap samples of the data.

# Arguments
- `rf::RandomForest`: The RandomForest instance to be fitted.
"""
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
    predict_forest(rf::RandomForest, data::Matrix{T}; regression=false) -> Vector

Predicts the output for the given data using the RandomForest instance.

# Arguments
- `rf::RandomForest`: The RandomForest instance used for prediction.
- `data::Matrix{T}`: The data matrix for which predictions are to be made.
- `regression::Bool`: A flag indicating whether to perform regression. Defaults to `false`.

# Returns
- A vector containing the predictions for each sample in the data.
"""
function predict_forest(rf::RandomForest, data::Matrix{T}, regression=false) where {T}    
    tree_predictions = [predict(tree, data) for tree in rf.trees]
    predictions = []

    for i in 1:size(data, 1)
        sample_predictions = [tree_predictions[j][i] for j in 1:rf.n_trees]
        if regression
            push!(predictions, mean(sample_predictions))
        else
            push!(predictions, mode(sample_predictions))
        end
    end
    return predictions
end


# Overwrite the find_best_split function to include max_features for gini imourity
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


# Override the build_tree function to include max_features parameter
function build_rf_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, max_features::Int, depth::Int=0) where {T, L}
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

function fit_rf_tree(tree::ClassificationTree, max_features)
    tree.root = build_rf_tree(tree.data, tree.labels, tree.max_depth, tree.min_samples_split, max_features)
end