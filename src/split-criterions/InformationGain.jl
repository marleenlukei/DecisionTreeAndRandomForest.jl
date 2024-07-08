
"""
    $(SIGNATURES)

Calculates the entropy of a vector of labels `y`.

## Arguments
- `y::AbstractVector`: A vector of labels.

## Returns
- `Float64`: The entropy of the vector.
"""
function calculate_entropy(y::AbstractVector)
    label_counts = countmap(y)
    probs = values(label_counts) ./ length(y)
    entropy = -sum(p -> p > 0 ? p * log2(p) : 0, probs)
    return entropy
end

"""
    $(SIGNATURES)

Calculate the Information Gain of a split.

## Arguments
- `y::AbstractVector{T}`: The original labels vector.
- `y_left::AbstractVector{T}`: The labels vector for the left split.
- `y_right::AbstractVector{T}`: The labels vector for the right split.

## Returns
- `Float64`: The Information Gain of the split.
"""
function weighted_entropy(y::T, y_left::T, y_right::T) where {T<:AbstractVector}
    H = calculate_entropy(y)
    H_left = calculate_entropy(y_left)
    H_right = calculate_entropy(y_right)
    p_left = length(y_left) / length(y)
    p_right = length(y_right) / length(y)
    return H - (p_left * H_left + p_right * H_right)
end


"""
    $(SIGNATURES)

Find the best split for the dataset `X` and labels `y` based on Information Gain.
Returns the best feature and threshold for the split.

## Arguments
- `X::AbstractMatrix`: A matrix of features.
- `y::AbstractVector`: A vector of labels.
- `num_features_to_use::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `best_feature::Int`: The index of the best feature to split on.
- `best_threshold::Real`: The threshold value for the best split.
"""
function information_gain(X::AbstractMatrix, y::AbstractVector, num_features_to_use::Int=-1)
    best_gain = -Inf
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
            gain = weighted_entropy(y, left_labels, right_labels)
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
- `data::AbstractMatrix`: A matrix of features, where each row is a data point and each column is a feature.
- `labels::AbstractVector`: A vector of labels corresponding to the data points.
- `num_features::Int`: The number of features to consider for each split.

## Returns
- `Tuple{Int, Real}`: A tuple containing the index of the best feature and the best split value.
"""
function split_ig(data::AbstractMatrix, labels::AbstractVector, num_features::Int=-1)
    return information_gain(data, labels, num_features)
end