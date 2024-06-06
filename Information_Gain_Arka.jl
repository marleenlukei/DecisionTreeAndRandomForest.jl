import Pkg; Pkg.add("Random")` 
using Random
using Statistics
using StatsBase  # Importing StatsBase for the countmap function

"""
    entropy(y::Vector)

Calculate the entropy of a vector of labels `y`.
"""
function entropy(y::Vector)
    counts = countmap(y)
    probs = values(counts) ./ length(y)
    return -sum(p -> p > 0 ? p * log2(p) : 0, probs)
end

"""
    information_gain(y::Vector, y_left::Vector, y_right::Vector)

Calculate the Information Gain of a split. `y` is the original labels vector, `y_left` and `y_right` are the labels vectors
for the left and right splits, respectively.
"""
function information_gain(y::Vector, y_left::Vector, y_right::Vector)
    H = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    p_left = length(y_left) / length(y)
    p_right = length(y_right) / length(y)
    return H - (p_left * H_left + p_right * H_right)
end

"""
    split_dataset(X::Matrix, y::Vector, feature::Int, threshold::Real)

Split the dataset `X` and labels `y` based on a `feature` and a `threshold`.
Returns the left and right splits for both `X` and `y`.
"""
function split_dataset(X::Matrix, y::Vector, feature::Int, threshold::Real)
    left_indices = findall(x -> x[feature] <= threshold, eachrow(X))
    right_indices = findall(x -> x[feature] > threshold, eachrow(X))
    X_left = X[left_indices, :]
    y_left = y[left_indices]
    X_right = X[right_indices, :]
    y_right = y[right_indices]
    return X_left, y_left, X_right, y_right
end

"""
    best_split(X::Matrix, y::Vector)

Find the best split for the dataset `X` and labels `y` based on Information Gain.
Returns the best feature and threshold for the split.
"""
function best_split(X::Matrix, y::Vector)
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
