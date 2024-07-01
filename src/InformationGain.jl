using StatsBase: countmap
using DocStringExtensions

"""
    $(SIGNATURES)

Calculates the entropy of a vector of labels `y`.

## Arguments
- `y::Vector{T}`: A vector of labels.

## Returns
- `Float64`: The entropy of the vector.
"""
function entropy(y::Vector{T}) where T
    counts = countmap(y)
    probs = values(counts) ./ length(y)
    return -sum(p -> p > 0 ? p * log2(p) : 0, probs)
end

"""
    $(SIGNATURES)

Calculate the Information Gain of a split.

## Arguments
- `y::Vector{T}`: The original labels vector.
- `y_left::Vector{T}`: The labels vector for the left split.
- `y_right::Vector{T}`: The labels vector for the right split.

## Returns
- `Float64`: The Information Gain of the split.
"""
function information_gain(y::Vector{T}, y_left::Vector{T}, y_right::Vector{T}) where T
    H = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    p_left = length(y_left) / length(y)
    p_right = length(y_right) / length(y)
    return H - (p_left * H_left + p_right * H_right)
end

"""
    $(SIGNATURES)

Split the dataset `X` and labels `y` based on a `feature` and a `threshold`.
Returns the left and right splits for both `X` and `y`.

## Arguments
- `X::AbstractMatrix{T}`: A matrix of features.
- `y::Vector{L}`: A vector of labels.
- `feature::Int`: The index of the feature to split on.
- `threshold::Real`: The threshold value to split the feature.

## Returns
- `X_left::AbstractMatrix{T}`, `y_left::Vector{L}`: The left split of the dataset and labels.
- `X_right::AbstractMatrix{T}`, `y_right::Vector{L}`: The right split of the dataset and labels.
"""
function split_dataset(X::AbstractMatrix{T}, y::Vector{L}, feature::Int, threshold::Real) where {T, L}
    left_indices = findall(x -> x[feature] <= threshold, eachrow(X))
    right_indices = findall(x -> x[feature] > threshold, eachrow(X))
    X_left = X[left_indices, :]
    y_left = y[left_indices]
    X_right = X[right_indices, :]
    y_right = y[right_indices]
    return X_left, y_left, X_right, y_right
end

"""
    $(SIGNATURES)

Find the best split for the dataset `X` and labels `y` based on Information Gain.
Returns the best feature and threshold for the split.

## Arguments
- `X::AbstractMatrix{T}`: A matrix of features.
- `y::Vector{L}`: A vector of labels.
- `num_features_to_use::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `best_feature::Int`: The index of the best feature to split on.
- `best_threshold::Real`: The threshold value for the best split.
"""
function best_split(X::AbstractMatrix{T}, y::Vector{L}, num_features_to_use::Int=-1) where {T, L}
    best_gain = -Inf
    best_feature = 0
    best_threshold = 0.0
    n_features = size(X, 2)
    features_to_use = 1:n_features
    if (num_features_to_use != -1)
        features_to_use = sample(1:n_features, num_features_to_use, replace=false)
    end
    for feature in features_to_use
        thresholds = unique(X[:, feature])
        for threshold in thresholds
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
            if length(y_left) == 0 || length(y_right) == 0
                continue
            end
            gain = information_gain(y, y_left, y_right)
            if gain > best_gain
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    return best_feature, best_threshold
end

"""
    $(SIGNATURES)

This function is a wrapper for `best_split` to be used as the split criterion in the `build_tree` function.

## Arguments
- `data::Matrix{T}`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::Vector{L}`: A vector of labels corresponding to the data points.
- `num_features::Int`: The number of features to consider for each split.

## Returns
- `Tuple{Int, Any}`: A tuple containing the index of the best feature and the best split value.
"""
function split_ig(data, labels, num_features)
    return best_split(data, labels, num_features)
end