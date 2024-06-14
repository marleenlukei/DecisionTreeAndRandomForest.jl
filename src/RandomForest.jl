using StatsBase: mode

"""
    RandomForest(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, number_of_trees::Int)
    RandomForest(data::Matrix{T}, labels::Vector{L}, number_of_trees::Int)
    RandomForest(data::Matrix{T}, labels::Vector{L})

Represents a RandomForest.

`trees` is the vector of ClassificationTree structures.
"""
struct RandomForest{T, L}
    trees::Vector{ClassificationTree{T, L}}

    function RandomForest(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int ,number_of_trees::Int) where {T, L}
        trees = Array{ClassificationTree{T, L}}(undef, number_of_trees)
        # Create n ClassificationTrees and save them in trees
        for i in 1:number_of_trees
            t = ClassificationTree(max_depth, min_samples_split, data, labels)
            trees[i] = t
        end
        new{T, L}(trees)
    end
end

RandomForest(data::Matrix{T}, labels::Vector{L}) where {T, L} = RandomForest(data, labels, -1, 1, 10)
RandomForest(data::Matrix{T}, labels::Vector{L}, number_of_trees::Int) where {T, L} = RandomForest(data, labels, -1, 1, number_of_trees)

"""
    fit(forest::RandomForest)

Trains a RandomForest.

`forest` is the RandomForest to be trained.
"""
function fit(forest::RandomForest)
    # Train every tree in forest.trees
    for tree in forest.trees
        fit(tree, true)
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

"""
    print_forest(forest::RandomForest)

Prints the structure of the RandomForest.

`forest` is the RandomForest to be printed.
"""
function print_forest(forest::RandomForest)
    for (index, tree) in enumerate(forest.trees)
        println("Tree $index")
        print_tree(tree)
    end
end