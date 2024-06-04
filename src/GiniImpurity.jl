using StatsBase

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

function weighted_gini(left_dataset::Vector{L}, right_dataset::Vector{L}) where {L}
    numberOfLeftRows = length(left_dataset)
    numberOfRightRows = length(right_dataset)
    total = numberOfLeftRows + numberOfRightRows
    gini_split = (numberOfLeftRows / total) * gini_impurity(left_dataset) + (numberOfRightRows / total) * gini_impurity(right_dataset)
    return gini_split
  end

  function split_node(data::Matrix{T}, labels::Vector{L}, index, value) where {T, L} 
    x_index = data[:, index]
    # if feature is numerical
    if eltype(x_index) <: Number
      mask = x_index .>= value
    # if feature is categorical
    else
      mask = x_index .== value
    end
    left = (labels[.!mask])
    right = (labels[mask])
    return left, right
  end


function find_best_split(data::Matrix{T}, labels::Vector{L}) where {T, L}  
    best_gini = Inf  
    best_feature_index = -1  
    best_feature_value = -1  
    num_features = size(data, 2)
    for feature_index in 1:num_features  
        unique_value = data[:, feature_index]  
        unique_value = unique(unique_value)  
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
    return best_feature_index, best_feature_value
end  






