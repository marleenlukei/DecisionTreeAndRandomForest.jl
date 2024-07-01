using StatsBase: mode, sample
using DocStringExtensions


"""
    $(SIGNATURES)

Represents a RandomForest.

## Fields
$(TYPEDFIELDS)
"""
struct RandomForest{T, L}
    "Contains the vector of ClassificationTree structures."
    trees::Vector{ClassificationTree{T, L}}
    "Contains the number of features to use when finding the best split. If -1, all the features are used."
    num_features::Int

    function RandomForest(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int , split_criterion::Function, number_of_trees::Int, subsample_percentage::Float64, num_features::Int) where {T, L}
        trees = Array{ClassificationTree{T, L}}(undef, number_of_trees)
        # Create n ClassificationTrees and save them in trees
        for i in 1:number_of_trees
            subsample_length = round(Int, size(data, 1) * subsample_percentage)
            subsample_idx = sample(1:size(data, 1), subsample_length, replace=true)
            t = ClassificationTree(max_depth, min_samples_split, split_criterion, data[subsample_idx, :], labels[subsample_idx])
            trees[i] = t
        end
        new{T, L}(trees, num_features)
    end
end

RandomForest(data::Matrix{T}, labels::Vector{L}, split_criterion::Function) where {T, L} = RandomForest(data, labels, -1, 1, split_criterion, 10, 0.8, -1)
RandomForest(data::Matrix{T}, labels::Vector{L}, split_criterion::Function, number_of_trees::Int) where {T, L} = RandomForest(data, labels, -1, 1, split_criterion, number_of_trees, 0.8, -1)
RandomForest(data::Matrix{T}, labels::Vector{L}, split_criterion::Function, number_of_trees::Int, subsample_percentage::Float64, num_features::Int) where {T, L} = RandomForest(data, labels, -1, 1, split_criterion, number_of_trees, subsample_percentage, num_features)

"""
    $(SIGNATURES)

This function trains each individual tree in the `RandomForest` by calling the `fit` function on each `ClassificationTree` within the `forest.trees` vector. The `num_features` parameter from the `RandomForest` object is used to control the number of features considered for each split during training.

## Arguments
- `forest::RandomForest`: The RandomForest to be trained.

## Returns
- `Nothing`: This function modifies the `forest` in-place.
"""
function fit(forest::RandomForest)
    # Train every tree in forest.trees
    for tree in forest.trees
        fit(tree, forest.num_features)
    end
end

"""
    $(SIGNATURES)

This function predicts the labels for each datapoint in the input dataset by using the trained `RandomForest`. 
Currently, it makes predictions using each individual tree in the forest and then combines the predictions using the most frequent label for each datapoint (Classification Task).

## Arguments
- `forest::RandomForest`: The trained RandomForest.
- `data::Matrix{T}`: The dataset for which to make predictions.

## Returns
- `Vector`: A vector of predictions for each datapoint in `data`.
"""
function predict(forest::RandomForest, data::Matrix{T}) where {T}
    # create a matrix to store the labels in
    labels = Matrix(undef, length(forest.trees), size(data, 1))
    for (index, tree) in enumerate(forest.trees)
        # Compute the prediction of every tree
        labels_for_tree = predict(tree, data)
        labels[index, :] = labels_for_tree
    end
    # Calculate the mode of every sample
    return [mode(labels[:, i]) for i in 1:size(labels, 2)]
end

"""
    $(SIGNATURES)

Prints the structure of the RandomForest.

## Arguments
- `forest::RandomForest`: The RandomForest to be printed.

## Returns
- `Nothing`: This function prints the structure of the `RandomForest`.
"""
function print_forest(forest::RandomForest)
    for (index, tree) in enumerate(forest.trees)
        println("Tree $index")
        print_tree(tree)
    end
end