
"""
    $(SIGNATURES)

This function calculates the Gini impurity of a set of labels, which measures the homogeneity of the labels within a node. A lower Gini impurity indicates a more homogeneous set of labels.

## Arguments
- `y::AbstractVector`: A vector of labels.

## Returns
- `Float64`: The Gini impurity of the labels.
"""
function calculate_gini(y::AbstractVector)
    label_counts = countmap(y)  
    label_probabilities = values(label_counts) ./ length(y)
    gini = 1 - sum(label_probabilities .^ 2)
    return gini
end

"""
    $(SIGNATURES)

This function calculates the weighted Gini impurity of a split, which is a measure of the impurity of the split considering the size of each subset. It's used to evaluate the quality of a split in a decision tree.

## Arguments
- `y_left::AbstractVector{T}`: A vector of labels for the left subset of the data.
- `y_right::AbstractVector{T}`: A vector of labels for the right subset of the data.

## Returns
- `Float64`: The weighted Gini impurity of the split.
"""
function weighted_gini(y_left::T, y_right::T) where {T<:AbstractVector}
    G_left = calculate_gini(y_left)
    G_right = calculate_gini(y_right)
    p_left = length(y_left) / (length(y_left) + length(y_right))
    p_right = length(y_right) / (length(y_left) + length(y_right))
    gini_impurity = (p_left * G_left + p_right * G_right)
    return gini_impurity
end


"""
    $(SIGNATURES)

Finds the best split point for a decision tree node.
For now it uses the Gini impurity as splitting criterion, but should later be extended to support other criteria.

## Arguments
- `X::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `y::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features_to_use::Int`: The number of features to consider when looking for the best split. If -1, all features are considered.

## Returns
- `Tuple{Int, T}`: A tuple containing the index of the best feature and the best split value.
"""
function gini_impurity(X::AbstractMatrix, y::AbstractVector, num_features_to_use::Int=-1)
    best_gini = Inf
    best_feature = -1
    best_threshold = -1
    n_features = size(X, 2)
    features_to_use = 1:n_features
    if (num_features_to_use != -1)
        features_to_use = sample(1:n_features, num_features_to_use, replace=false)
    end
    for feature in features_to_use
        thresholds = unique(X[:, feature])

        for threshold in thresholds
            left_labels, right_labels = split_node(X, y, feature, threshold)
            gini = weighted_gini(left_labels, right_labels)
            if gini < best_gini
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    return (best_feature, best_threshold)
end

"""
    $(SIGNATURES)

This function is a wrapper for `find_best_split` to be used as the split criterion in the `build_tree` function.

## Arguments
- `X::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `y::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features::Int`: The number of features to consider when looking for the best split. If -1, all features are considered.

## Returns
- `Tuple{Int, T}`: A tuple containing the index of the best feature and the best split value.
"""
function split_gini(X::AbstractMatrix, y::AbstractVector, num_features::Int=-1)
    return gini_impurity(X, y, num_features)
end





