
"""
    $(SIGNATURES)

Represents a RandomForest.

## Fields
$(TYPEDFIELDS)
"""
struct RandomForest
    "Contains the vector of DecisionTree structures."
    trees::Vector{DecisionTree}
    "Contains the maximum depth of the tree. If -1, the DecisionTree is of unlimited depth."
    max_depth::Int
    "Contains the minimum number of samples required to split a node."
    min_samples_split::Int
    "Contains the split criterion function."
    split_criterion::Function
    "Contains the number of trees in the RandomForest structure."
    number_of_trees::Int
    "Contains the percentage of the dataset to use for training each tree."
    subsample_percentage::Float64
    "Contains the number of features to use when finding the best split. If -1, all the features are used."
    num_features::Int

    function RandomForest(max_depth::Int, min_samples_split::Int, split_criterion::Function, number_of_trees::Int, subsample_percentage::Float64, num_features::Int)
        new([], max_depth, min_samples_split, split_criterion, number_of_trees, subsample_percentage, num_features)
    end
end


RandomForest(split_criterion::Function) = RandomForest(-1, 1, split_criterion, 10, 0.8, -1)
RandomForest(split_criterion::Function, number_of_trees::Int) = RandomForest(-1, 1, split_criterion, number_of_trees, 0.8, -1)
RandomForest(split_criterion::Function, number_of_trees::Int, subsample_percentage::Float64, num_features::Int) = RandomForest(-1, 1, split_criterion, number_of_trees, subsample_percentage, num_features)

"""
    $(SIGNATURES)

This function trains each individual tree in the `RandomForest` by calling the `fit` function on each `ClassificationTree` within the `forest.trees` vector. The `num_features` parameter from the `RandomForest` object is used to control the number of features considered for each split during training.

## Arguments
- `forest::RandomForest`: The RandomForest to be trained.

## Returns
- `Nothing`: This function modifies the `forest` in-place.
"""
function fit!(forest::RandomForest, data::AbstractMatrix, labels::AbstractVector)
    for _ in 1:forest.number_of_trees
        subsample_length = round(Int, size(data, 1) * forest.subsample_percentage)
        subsample_idx = sample(1:size(data, 1), subsample_length, replace=true)
        tree = DecisionTree(forest.max_depth, forest.min_samples_split, forest.num_features, forest.split_criterion)
        fit!(tree, data[subsample_idx, :], labels[subsample_idx])
        push!(forest.trees, tree)
    end
end

"""
    $(SIGNATURES)

This function predicts the labels for each datapoint in the input dataset by using the trained `RandomForest`. 
Currently, it makes predictions using each individual tree in the forest and then combines the predictions using the most frequent label for each datapoint (Classification Task).

## Arguments
- `forest::RandomForest`: The trained RandomForest.
- `data::AbstractMatrix`: The dataset for which to make predictions.

## Returns
- `AbstractVector`: A vector of predictions for each datapoint in `data`.
"""
function predict(forest::RandomForest, data::AbstractMatrix)
    # create a matrix to store the labels in
    labels = Matrix(undef, length(forest.trees), size(data, 1))
    for (index, tree) in enumerate(forest.trees)
        # Compute the prediction of every tree
        labels_for_tree = predict(tree, data)
        labels[index, :] = labels_for_tree
    end
    # Calculate the mode of every sample
    is_regression = eltype(identity.(labels)) <: Number
    if is_regression
        return [mean(col) for col in eachcol(labels)]
    else
        return [mode(col) for col in eachcol(labels)]
    end
end

"""
    $(SIGNATURES)

This function recursively prints the structure of the ForestTree. It's primarily used for debugging and visualizing the Forest structure.

## Arguments
- `io::IO`: The IO context to print the Forest structure.
- `forest::RandomForest`: The RandomForest to be printed.

## Returns
- `Nothing`: This function prints the structure of the `RandomForest`.
"""
function Base.show(io::IO, forest::RandomForest)
    for (index, tree) in enumerate(forest.trees)
        println(io, "Tree $index")
        print(tree)
        println(io, "Tree $index")
        print(tree)
    end
end