using StatsBase: mode, sample

"""
Represents a single decision tree used in the Random Forest.

`max_depth` controls the maximum depth of the tree. If -1, the depth is not limited.

`min_samples_split` controls when a node in the decision tree should be split.

`max_features` controls the number of features to consider when looking for the best split.

`root` contains the root node of the tree.
"""
mutable struct DecisionTree{T, L}
    max_depth::Int
    min_samples_split::Int
    max_features::Int

    root::Union{Node{T, L}, Leaf{L}, Missing}

    DecisionTree(max_depth::Int, min_samples_split::Int, max_features::Int) where {T, L} =
        new{T, L}(max_depth, min_samples_split, max_features, missing)
end

"""
Represents a Random Forest for classification.

`n_trees` is the number of trees in the forest.

`max_depth` controls the maximum depth of each tree. If -1, the depth is not limited.

`min_samples_split` controls when a node in the decision tree should be split.

`max_features` controls the number of features to consider when looking for the best split.

`trees` is a vector containing the individual DecisionTree instances.
"""
mutable struct RandomForest{T, L}
    n_trees::Int
    max_depth::Int
    min_samples_split::Int
    max_features::Int

    trees::Vector{DecisionTree{T, L}}

    RandomForest(n_trees::Int, max_depth::Int, min_samples_split::Int, max_features::Int) where {T, L} =
        new{T, L}(n_trees, max_depth, min_samples_split, max_features, DecisionTree{T, L}[])
end

"""
    fit_tree(tree::DecisionTree, data::Matrix{T}, labels::Vector{L})

Build the tree structure for the given DecisionTree.
"""
function fit_tree(tree::DecisionTree{T, L}, data::Matrix{T}, labels::Vector{L}) where {T, L}
    tree.root = build_tree(data, labels, tree.max_depth, tree.min_samples_split, 0, tree.max_features)
end

"""function fit_forest(rf::RandomForest)
for i in 1:rf.n_trees
    
    # tree = ClassificationTree(rf.max_depth, rf.min_samples_split, rf.data, rf.labels)
    
    # Sample data with replacement (bootstrapping)
    indices = sample(1:size(rf.data, 1), size(rf.data, 1), replace=true)
    bootstrap_data = rf.data[indices, :]
    bootstrap_labels = rf.labels[indices]

    # Build the tree with the bootstrapped data
    # rf.trees = vcat(rf.trees, build_tree(bootstrap_data, bootstrap_labels, rf.max_depth, rf.min_samples_split, rf.max_features))

    tree = ClassificationTree(rf.max_depth, rf.min_samples_split, bootstrap_data, bootstrap_labels)

    # Build_tree has to be adjusted to take max_features as parameters
    fit_rf_tree(tree, rf.max_features)
    
    push!(rf.trees, tree)

end
end"""


"""
    fit_forest(forest::RandomForest, data::Matrix{T}, labels::Vector{L})

Build the forest by training each DecisionTree on a bootstrap sample of the data.
"""
function fit_forest(forest::RandomForest{T, L}, data::Matrix{T}, labels::Vector{L}) where {T, L}
    for i in 1:forest.n_trees
        tree = DecisionTree{T, L}(forest.max_depth, forest.min_samples_split, forest.max_features)

        # Sample data with replacement (bootstrapping)
        indices = sample(1:size(data, 1), size(data, 1), replace=true)
        bootstrap_data = data[indices, :]
        bootstrap_labels = labels[indices]

        fit_tree(tree, bootstrap_data, bootstrap_labels)
        push!(forest.trees, tree)
    end
end

"""
    predict_tree(tree::DecisionTree, data::Matrix{T})

Predict labels for the given data using the trained DecisionTree.
"""
function predict_tree(tree::DecisionTree{T, L}, data::Matrix{T}) where {T, L}
    predictions = []
    
    for i in 1:size(data, 1)
        node = tree.root
        while !isa(node, Leaf)
            if isa(node.split_value, Number)
                if data[i, node.feature_index] < node.split_value
                    node = node.left
                else
                    node = node.right
                end
            else
                if data[i, node.feature_index] != node.split_value
                    node = node.left
                else
                    node = node.right
                end
            end
        end
        # Get the label that occurs the most and add it to predictions
        push!(predictions, mode(node.values))
    end
    return predictions
end

"""
    predict_forest(forest::RandomForest, data::Matrix{T})

Predict labels for the given data using the trained RandomForest.
"""
function predict_forest(forest::RandomForest{T, L}, data::Matrix{T}) where {T, L}
    tree_predictions = [predict_tree(tree, data) for tree in forest.trees]
    predictions = []

    for i in 1:size(data, 1)
        sample_predictions = [tree_predictions[j][i] for j in 1:forest.n_trees]
        push!(predictions, mode(sample_predictions))
    end
    
    return predictions
end

"""
Helper function to find the best split considering only a subset of features.
"""
function find_best_split(data::Matrix{T}, labels::Vector{L}, max_features::Int) where {T, L}
    # Select a random subset of features
    features = sample(1:size(data, 2), max_features, replace=false)

    best_feature = -1
    best_split_value = missing
    best_gini = Inf

    for feature in features
        unique_values = unique(data[:, feature])
        for value in unique_values
            left_mask = data[:, feature] .< value
            right_mask = data[:, feature] .>= value

            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            if length(left_labels) == 0 || length(right_labels) == 0
                continue
            end

            left_gini = gini_impurity(left_labels)
            right_gini = gini_impurity(right_labels)
            gini = (length(left_labels) * left_gini + length(right_labels) * right_gini) / length(labels)

            if gini < best_gini
                best_gini = gini
                best_feature = feature
                best_split_value = value
            end
        end
    end

    return best_feature, best_split_value
end

"""
Helper function to calculate the Gini impurity of a set of labels.
"""
function gini_impurity(labels::Vector{L}) where {L}
    n = length(labels)
    _, counts = countmap(labels)
    impurity = 1.0
    for count in counts
        prob = count / n
        impurity -= prob^2
    end
    return impurity
end

"""
Override the build_tree function to include max_features parameter.
"""
function build_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, depth::Int, max_features::Int) where {T, L}
    # If max_depth is reached or if the data can not be split further, return a leaf
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth)
        return Leaf{L}(labels)
    end

    # if all the labels are the same, return a leaf
    if allequal(labels)
        return Leaf{L}(labels)
    end

    # Get the best split from the respective split_criterion
    feature_index, split_value = find_best_split(data, labels, max_features)

    # Create the mask on the data
    if isa(split_value, Number)
        left_mask = data[:, feature_index] .< split_value
        right_mask = data[:, feature_index] .>= split_value
    else
        left_mask = data[:, feature_index] .!= split_value
        right_mask = data[:, feature_index] .== split_value
    end

    # Compute the data and labels for the child nodes
    left_data, right_data = data[left_mask, :], data[right_mask, :]
    left_labels, right_labels = labels[left_mask], labels[right_mask]

    # Create the node with the computed attributes
    node = Node(data, labels)
    node.feature_index = feature_index
    node.split_value = split_value

    # Build the subtrees for the node
    node.left = build_tree(left_data, left_labels, max_depth, min_samples_split, depth + 1, max_features)
    node.right = build_tree(right_data, right_labels, max_depth, min_samples_split, depth + 1, max_features)

    # Return the node
    return node
end

# Example usage
data = [ ... ]  # your data matrix
labels = [ ... ]  # your labels vector

# Create and build a RandomForest
rf = RandomForest{Float64, Int}(100, -1, 2, 5)
fit_forest(rf, data, labels)

# Predict with the RandomForest
predictions = predict_forest(rf, data)
println(predictions)
