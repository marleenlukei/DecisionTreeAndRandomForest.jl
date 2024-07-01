using StatsBase: countmap, sample
using DocStringExtensions

"""
    $(SIGNATURES)

This function calculates the Gini impurity of a set of labels, which measures the homogeneity of the labels within a node. A lower Gini impurity indicates a more homogeneous set of labels.

## Arguments
- `labels::Vector{L}`: A vector of labels.

## Returns
- `Float64`: The Gini impurity of the labels.
"""
function gini_impurity(labels::Vector{L}) where {L}

    label_counts_dict = countmap(labels)  #dict with count per label
    total = length(labels)
    if total == 0  
        return 0.0  
    end 
    label_probabilities = values(label_counts_dict) ./ total
    gini = 1 - sum(label_probabilities.^2)
    return gini
end

"""
    $(SIGNATURES)

This function calculates the weighted Gini impurity of a split, which is a measure of the impurity of the split considering the size of each subset. It's used to evaluate the quality of a split in a decision tree.

## Arguments
- `left_dataset::Vector{L}`: A vector of labels for the left subset of the data.
- `right_dataset::Vector{L}`: A vector of labels for the right subset of the data.

## Returns
- `Float64`: The weighted Gini impurity of the split.
"""
function weighted_gini(left_dataset::Vector{L}, right_dataset::Vector{L}) where {L}
    number_of_left_rows = length(left_dataset)
    number_of_right_rows = length(right_dataset)
    total = number_of_left_rows + number_of_right_rows
    gini_split = (number_of_left_rows / total) * gini_impurity(left_dataset) + (number_of_right_rows / total) * gini_impurity(right_dataset)
    return gini_split
  end

"""
  $(SIGNATURES)

This function splits the labels into two subsets based on the provided feature and value. It handles both numerical and categorical features.

## Arguments
- `data::Matrix{T}`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::Vector{L}`: A vector of labels corresponding to the data points.
- `index::Int`: The index of the feature to split on.
- `value`: The value to split the feature on.

## Returns
- `Tuple{Vector{L}, Vector{L}}`: A tuple containing the left and right sets of labels.
"""
function split_node(data::Matrix{T}, labels::Vector{L}, index, value) where {T, L} 
    x_index = data[:, index]
    # if feature is numerical
    if eltype(identity.(x_index)) <: Number
      mask = x_index .>= value
    # if feature is categorical
    else
      mask = x_index .== value
    end
    left = (labels[.!mask])
    right = (labels[mask])
    return left, right
end

"""
    $(SIGNATURES)

Finds the best split point for a decision tree node.
For now it uses the Gini impurity as splitting criterion, but should later be extended to support other criteria.

## Arguments
- `data::Matrix{T}`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::Vector{L}`: A vector of labels corresponding to the data points.
- `num_features_to_use::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `Tuple{Int, Any}`: A tuple containing the index of the best feature and the best split value.
"""
function find_best_split(data::Matrix{T}, labels::Vector{L}, num_features_to_use::Int=-1) where {T, L}  
    best_gini = Inf  
    best_feature_index = -1  
    best_feature_value = -1  
    num_features = size(data, 2)
    features_to_use = 1:num_features
    if (num_features_to_use != -1)
        features_to_use = sample(1:num_features, num_features_to_use, replace=false)
    end
    for feature_index in features_to_use 
        value = data[:, feature_index]  
        unique_value = unique(value)  
        for value in unique_value  
            left_labels,right_labels = split_node(data, labels, feature_index, value) 
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

"""
    $(SIGNATURES)

This function is a wrapper for `find_best_split` to be used as the split criterion in the `build_tree` function.

## Arguments
- `data::Matrix{T}`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::Vector{L}`: A vector of labels corresponding to the data points.
- `num_features::Int`: The number of features to consider for each split.

## Returns
- `Tuple{Int, Any}`: A tuple containing the index of the best feature and the best split value.
"""
function split_gini(data, labels, num_features)
    return find_best_split(data, labels, num_features)
end





