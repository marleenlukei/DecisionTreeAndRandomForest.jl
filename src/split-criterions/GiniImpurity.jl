"""
Calculates the Gini impurity of a set of labels.

Args:
    labels: A vector of labels.

Returns:
    The Gini impurity of the labels.
"""
function gini_impurity(labels::AbstractVector)

    label_counts_dict = countmap(labels)  #dict with count per label
    total = length(labels)
    if total == 0
        return 0.0
    end
    label_probabilities = values(label_counts_dict) ./ total
    gini = 1 - sum(label_probabilities .^ 2)
    return gini
end

"""
Calculates the weighted Gini impurity of a split.

Args:
    left_dataset: A vector of labels for the left subset of the data.
    right_dataset: A vector of labels for the right subset of the data.

Returns:
    The weighted Gini impurity of the split.
"""
function weighted_gini(left_dataset::T, right_dataset::T) where {T<:AbstractVector}

    number_of_left_rows = length(left_dataset)
    number_of_right_rows = length(right_dataset)
    total = number_of_left_rows + number_of_right_rows
    gini_split = (number_of_left_rows / total) * gini_impurity(left_dataset) + (number_of_right_rows / total) * gini_impurity(right_dataset)
    return gini_split
end

"""
Splits the labels into two nodes based on the provided feature and value.

Args:
    data: A matrix of features, where each row is a data point and each column is a feature.
    labels: A vector of labels corresponding to the data points.
    index: The index of the feature to split on.
    value: The value to split the feature on.

Returns:
    A tuple containing the left and right sets of labels.
  """
function split_node(data::AbstractMatrix{T}, labels::AbstractVector, index::Int, value::T) where {T}
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
Finds the best split point for a decision tree node.
For now it uses the Gini impurity as splitting criterion, but should later be extended to support other criteria.

Args:
    data: A matrix of features, where each row is a data point and each column is a feature.
    labels: A vector of labels corresponding to the data points.

Returns:
    A tuple containing the index of the best feature and the best split value.
"""
function find_best_split(data::AbstractMatrix, labels::AbstractVector, num_features_to_use::Int=-1)
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


function split_gini(data::AbstractMatrix, labels::AbstractVector, num_features::Int=-1)
    return find_best_split(data, labels, num_features)
end





