"""
    RandomForest(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, number_of_trees::Int, subsample_percentage::Float64)
    RandomForest(data::Matrix{T}, labels::Vector{L}, number_of_trees::Int, subsample_percentage::Float64)
    RandomForest(data::Matrix{T}, labels::Vector{L})

Represents a RandomForest.

`trees` is the vector of DecisionTree structures.
`num_features` is the number of features to use when finding the best split. If -1, all the features are used.
"""
struct RandomForest
    trees::Vector{DecisionTree}
    max_depth::Int
    min_samples_split::Int 
    split_criterion::Function
    number_of_trees::Int
    subsample_percentage::Float64
    num_features::Int

    function RandomForest(max_depth::Int, min_samples_split::Int , split_criterion::Function, number_of_trees::Int, subsample_percentage::Float64, num_features::Int)
        new([], max_depth, min_samples_split, split_criterion, number_of_trees, subsample_percentage, num_features)
    end
end

RandomForest(split_criterion::Function) = RandomForest(-1, 1, split_criterion, 10, 0.8, -1)
RandomForest(split_criterion::Function, number_of_trees::Int) = RandomForest(-1, 1, split_criterion, number_of_trees, 0.8, -1)
RandomForest(split_criterion::Function, number_of_trees::Int, subsample_percentage::Float64, num_features::Int) = RandomForest(-1, 1, split_criterion, number_of_trees, subsample_percentage, num_features)
"""
    fit(forest::RandomForest)

Trains a RandomForest.

`forest` is the RandomForest to be trained.
`num_features` is the number of features to use when finding the best split.
"""
function fit(forest::RandomForest, data::Matrix, labels::Vector)
    for _ in 1:forest.number_of_trees
        subsample_length = round(Int, size(data, 1) * forest.subsample_percentage)
        subsample_idx = sample(1:size(data, 1), subsample_length, replace=true)
        tree = DecisionTree(forest.max_depth, forest.min_samples_split, forest.num_features, forest.split_criterion)
        fit(tree, data[subsample_idx, :], labels[subsample_idx])
        push!(forest.trees, tree)
    end
end

"""
    predict(forest::RandomForest, data::Matrix{T})

Predicts the labels for the samples in `data`.

`forest` is the RandomForest used to predict the labels.

`data` contains the samples to predict the labels of.
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

function Base.show(io::IO, forest::RandomForest)
    for (index, tree) in enumerate(forest.trees)
        println(io, "Tree $index")
        print(tree)
    end
end