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


#should later be generalized to use different splitting criteria  
function find_best_split(data::Matrix{T}, labels::Vector{L}) where {T, L}  
    best_gini = Inf  
    best_feature_index = -1  
    best_feature_value = -1  

    num_features = size(data, 2)  
  
    for feature_index in 1:num_features  
        feature_samples = data[:, feature_index]  
        unique_samples = unique(feature_samples)  
        for sample in unique_samples  
            left_indices = data[:, feature_index] .<= sample  
            right_indices = data[:, feature_index] .> sample  
            left_labels = labels[left_indices]  
            right_labels = labels[right_indices]  
            gini = weighted_gini(left_labels, right_labels)  
            if gini < best_gini  
                best_gini = gini  
                best_feature_index = feature_index  
                best_feature_value = sample 
            end  
        end  
    end  
    return best_feature_index, best_feature_value
end  



