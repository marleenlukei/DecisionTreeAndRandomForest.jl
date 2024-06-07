using Random
using Statistics
using StatsBase: countmap

"""
    entropy(y::Vector{T}) where T

Calculate the entropy of a vector of labels `y`.

# Arguments
- `y`: A vector of labels.

# Returns
- The entropy of the vector.
"""
function entropy(y::Vector{T}) where T
    counts = countmap(y)
    probs = values(counts) ./ length(y)
    return -sum(p -> p > 0 ? p * log2(p) : 0, probs)
end

"""
    information_gain(y::Vector{T}, y_left::Vector{T}, y_right::Vector{T}) where T

Calculate the Information Gain of a split.

# Arguments
- `y`: The original labels vector.
- `y_left`: The labels vector for the left split.
- `y_right`: The labels vector for the right split.

# Returns
- The Information Gain of the split.
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
    split_dataset(X::AbstractMatrix{T}, y::Vector{T}, feature::Int, threshold::Real) where T

Split the dataset `X` and labels `y` based on a `feature` and a `threshold`.
Returns the left and right splits for both `X` and `y`.

# Arguments
- `X`: A matrix of features.
- `y`: A vector of labels.
- `feature`: The index of the feature to split on.
- `threshold`: The threshold value to split the feature.

# Returns
- `X_left`, `y_left`: The left split of the dataset and labels.
- `X_right`, `y_right`: The right split of the dataset and labels.
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
    best_split(X::AbstractMatrix{T}, y::Vector{T}) where T

Find the best split for the dataset `X` and labels `y` based on Information Gain.
Returns the best feature and threshold for the split.

# Arguments
- `X`: A matrix of features.
- `y`: A vector of labels.

# Returns
- `best_feature`: The index of the best feature to split on.
- `best_threshold`: The threshold value for the best split.
"""
function best_split(X::AbstractMatrix{T}, y::Vector{L}) where {T, L}
    best_gain = -Inf
    best_feature = -1
    best_threshold = 0.0
    n_features = size(X, 2)
    
    for feature in 1:n_features
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
