"""
    $(SIGNATURES)

Calculate the sample variance of a given set of labels. It uses the standard formula for sample variance.

## Arguments
- `y::AbstractVector`: A vector of numerical labels for which the variance is to be computed.

## Returns
- `Float64`: The sample variance of the input label vector `y`.
"""
function calculate_variance(y::AbstractVector)
    variance = sum((y .- mean(y)) .^ 2) / length(y) - 1
    return variance
end

"""
    $(SIGNATURES)

Calculates the variance reduction achieved by a split.

## Arguments
- `y_left::AbstractVector{T}`: A vector of labels for the left subset of the data.
- `y_right::AbstractVector{T}`: A vector of labels for the right subset of the data.

## Returns
- `Float64`: The variance reduction achieved by the split.
"""
function weighted_variance(y_left::T, y_right::T) where {T<:AbstractVector}
    V_left = calculate_variance(y_left)
    V_right = calculate_variance(y_right)
    p_left = length(y_left) / (length(y_left) + length(y_right))
    p_right = length(y_right) / (length(y_left) + length(y_right))
    variance_reduction = (p_left * V_left + p_right * V_right)
    return variance_reduction
end


"""
    $(SIGNATURES)

Finds the best split point for a decision tree node using variance reduction.

## Arguments
- `X::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `y::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features_to_use::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `Tuple{Int, Any}`: A tuple containing the index of the best feature and the best split value.
"""
function variance_reduction(X::AbstractMatrix, y::AbstractVector, num_features_to_use::Int=-1)
    best_variance = -Inf
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
            if length(left_labels) == 0 || length(right_labels) == 0
                continue
            end

            variance = calculate_variance(y) - weighted_variance(left_labels, right_labels)

            if variance > best_variance
                best_variance = variance
                best_feature = feature
                best_threshold = threshold
            end
        end
    end

    return best_feature, best_threshold
end

"""
This function is a wrapper for `find_best_split_vr` to be used as the split criterion in the `build_tree` function.

## Arguments
- `X::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `y::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features::Int`: The number of features to consider for each split.

## Returns
- `Tuple{Int, T}`: A tuple containing the index of the best feature and the best split value.
"""
function split_variance(X::AbstractMatrix, y::AbstractVector, num_features::Int=-1)
    return variance_reduction(X, y, num_features)
end