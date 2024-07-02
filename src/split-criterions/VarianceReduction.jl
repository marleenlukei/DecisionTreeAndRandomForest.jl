"""
    $(SIGNATURES)

Calculate the sample variance of a given set of labels. It uses the standard formula for sample variance.

## Arguments
- `y::AbstractVector`: A vector of numerical labels for which the variance is to be computed.

## Returns
- `Float64`: The sample variance of the input label vector `y`.
"""
function variance(y::AbstractVector)
    variance = sum((y .- mean(y)) .^ 2) / length(y) - 1
    return variance
end

"""
    $(SIGNATURES)

Calculates the variance reduction achieved by a split.

## Arguments
- `left_dataset::AbstractVector{T}`: A vector of labels for the left subset of the data.
- `right_dataset::AbstractVector{T}`: A vector of labels for the right subset of the data.

## Returns
- `Float64`: The variance reduction achieved by the split.
"""
function variance_reduction(left_dataset::T, right_dataset::T) where {T<:AbstractVector}
    number_of_left_rows = length(left_dataset)
    number_of_right_rows = length(right_dataset)
    total = number_of_left_rows + number_of_right_rows
    variance_reduction = ((number_of_left_rows / total) * variance(left_dataset) + (number_of_right_rows / total) * variance(right_dataset))
    return variance_reduction
end

"""
    $(SIGNATURES)

Splits the labels into two nodes based on the provided feature and value.

## Arguments
- `data::AbstractMatrix{T}`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::AbstractVector`: A vector of labels corresponding to the data points.
- `index::Int`: The index of the feature to split on.
- `value::T`: The value to split the feature on.

## Returns
- `Tuple{AbstractVector, AbstractVector}`: A tuple containing the left and right sets of labels.
  """
function split_node_vr(data::AbstractMatrix{T}, labels::AbstractVector, index::Int, value::T) where {T}
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

Finds the best split point for a decision tree node using variance reduction.

## Arguments
- `data::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features_to_use::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `Tuple{Int, Any}`: A tuple containing the index of the best feature and the best split value.
"""
function find_best_split_vr(data::AbstractMatrix, labels::AbstractVector, num_features_to_use::Int=-1)
    best_feature_index = -1
    best_threshold = -1
    best_variance = -Inf
    num_features = size(data, 2)
    features_to_use = 1:num_features
    if (num_features_to_use != -1)
        features_to_use = sample(1:num_features, num_features_to_use, replace=false)
    end

    for feature in features_to_use
        thresholds = unique(data[:, feature])
        for threshold in thresholds
            left_labels, right_labels = split_node_vr(data, labels, feature, threshold)
            if length(left_labels) == 0 || length(right_labels) == 0
                continue
            end

            current_variance = variance(labels) - variance_reduction(left_labels, right_labels)

            if current_variance > best_variance
                best_variance = current_variance
                best_feature_index = feature
                best_threshold = threshold
            end
        end
    end

    return best_feature_index, best_threshold
end

"""
This function is a wrapper for `find_best_split_vr` to be used as the split criterion in the `build_tree` function.

## Arguments
- `data::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features::Int`: The number of features to consider for each split.

## Returns
- `Tuple{Int, T}`: A tuple containing the index of the best feature and the best split value.
"""
function split_variance(data::AbstractMatrix, labels::AbstractVector, num_features::Int=-1)
    return find_best_split_vr(data, labels, num_features)
end