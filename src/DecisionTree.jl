"""
    $(SIGNATURES)

Represents a Leaf in the ClassificationTree structure.

## Fields
$(TYPEDFIELDS)
"""
mutable struct Leaf{T<:AbstractVector}
    values::T

    Leaf(values::T) where {T<:AbstractVector} = new{T}(values)
end

"""
    $(SIGNATURES)

Represents a Node in the ClassificationTree structure.

## Fields
$(TYPEDFIELDS)
"""
mutable struct Node
    "Points to the left child." 
    left::Union{Node,Leaf}
    "Points to the right child."
    right::Union{Node,Leaf}
    "Stores the index of the selected feature."
    feature_index::Int
    "Stores the value on that the data is split."
    split_value

    Node(left::Union{Node,Leaf}, right::Union{Node,Leaf}, feature_index::Int, split_value) = new(left, right, feature_index, split_value)
end

"""
    $(SIGNATURES)

Represents a ClassificationTree.

## Fields
$(TYPEDFIELDS)
"""
mutable struct DecisionTree
    "Controls the maximum depth of the tree. If -1, the DecisionTree is of unlimited depth."
    max_depth::Int
    "Controls the minimum number of samples required to split a node."
    min_samples_split::Int
    num_features::Int
    "Contains the split criterion function."
    split_criterion::Function
    "Contains the root node of the ClassificationTree."
    root::Union{Node,Leaf,Missing}

    DecisionTree(max_depth::Int, min_samples_split::Int, num_features::Int, split_criterion::Function) = new(max_depth, min_samples_split, num_features, split_criterion, missing)
end

DecisionTree(split_criterion::Function) = DecisionTree(-1, 1, -1, split_criterion)
DecisionTree(max_depth::Int, min_samples_split::Int, split_criterion::Function) = DecisionTree(max_depth, min_samples_split, -1, split_criterion)


"""
    $(SIGNATURES)

This function recursively builds a ClassificationTree by iteratively splitting the data based on the provided `split_criterion`. The process continues until either the maximum depth is reached, the number of samples in a node falls below `min_samples_split` or all labels in a node are the same.

## Arguments
- `data::Matrix{T}`: The training data.
- `labels::Vector{L}`: The labels for the training data.
- `max_depth::Int`: The maximum depth of the tree.
- `min_samples_split::Int`: The minimum number of samples required to split a node.
- `split_criterion::Function`: The function used to determine the best split at each node.
- `depth::Int=0`: The current depth of the tree (used recursively).
- `num_features::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `Union{Node{T, L}, Leaf{L}}`: The root node of the built tree.
"""
function build_tree(data::AbstractMatrix, labels::AbstractVector, max_depth::Int, min_samples_split::Int, num_features::Int, split_criterion::Function, depth::Int=0)
    # If max_depth is reached or if the data can not be split further, return a leaf
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth)
        return Leaf(labels)
    end

    # if all the labels are the same, return a leaf
    if allequal(labels)
        return Leaf(labels)
    end

    # Get the best split from the respective split_criterion
    feature_index, split_value = split_criterion(data, labels, num_features)

    if feature_index == -1
        return Leaf(labels)
    end
    # Create the mask on the data
    if isa(split_value, Number)
        mask = data[:, feature_index] .>= split_value
    else
        mask = data[:, feature_index] .== split_value
    end
    # Compute the data and labels for the child nodes
    left_data, right_data = data[.!mask, :], data[mask, :]
    left_labels, right_labels = labels[.!mask], labels[mask]

    # if one subtree would be empty return a Leaf
    if (length(left_labels) * length(right_labels) == 0)
        return Leaf(labels)
    end

    # Build the subtrees for the node
    left = build_tree(left_data, left_labels, max_depth, min_samples_split, num_features, split_criterion, depth + 1)
    right = build_tree(right_data, right_labels, max_depth, min_samples_split, num_features, split_criterion, depth + 1)
    # Create the node with the computed attributes
    node = Node(left, right, feature_index, split_value)

    # Return the node
    return node
end

"""
    $(SIGNATURES)

This function builds the tree structure of the `ClassificationTree` by calling the `build_tree` function. It uses the training data and parameters stored within the `tree` object.

## Arguments
- `tree::ClassificationTree`: The ClassificationTree to fit.
- `num_features::Int=-1`: The number of features to consider for each split. If -1, all features are used.

## Returns
- `Nothing`: This function modifies the `tree` in-place.
"""
function fit!(tree::DecisionTree, data::AbstractMatrix, labels::AbstractVector)
    if size(data, 1) != length(labels)
        throw(ArgumentError("The number of rows in data must match the number of elements in labels -> $(size(data, 1)) != $(length(labels))"))
    end
    tree.root = build_tree(data, labels, tree.max_depth, tree.min_samples_split, tree.num_features, tree.split_criterion, 0)
end

"""
    $(SIGNATURES)

This function traverses the tree structure of the `ClassificationTree` for each datapoint in `data`. It follows the decision rules based on the split criteria and feature values. If the leaf node contains numerical values, its treated as a regreesion problem and the prediction is the average of those values. If a leaf node contains numerical values, it is treated as a regression problem, and the prediction is the average of those values. If the leaf node contains categorical labels, it is treated as a classification problem, and the prediction is the most frequent label (mode) among the labels in the leaf node.

## Arguments
- `tree::ClassificationTree`: The trained ClassificationTree.
- `data::Matrix{T}`: The datapoints to predict.

## Returns
- `Vector`: A vector of predictions for each datapoint in `data`.
"""
function predict(tree::DecisionTree, data::AbstractMatrix)

    if isa(tree.root, Missing)
        throw(UndefVarError("The tree needs to be fitted first!"))
    end

    predictions = []

    for sample in eachrow(data)
        node = tree.root
        while !isa(node, Leaf)
            if isa(node.split_value, Number)
                if sample[node.feature_index] < node.split_value
                    node = node.left
                else
                    node = node.right
                end
            else
                if sample[node.feature_index] != node.split_value
                    node = node.left
                else
                    node = node.right
                end
            end
        end
        if all(isa.(node.values, Number))
            push!(predictions, mean(node.values))
        else
            push!(predictions, mode(node.values))
        end
    end
    return predictions
end

"""
    $(SIGNATURES)

This function recursively prints the structure of the `ClassificationTree`, providing information about each node and leaf. It's primarily used for debugging and visualizing the tree's structure.

## Arguments
- `tree::ClassificationTree`: The ClassificationTree to print.
"""
function Base.show(io::IO, tree::DecisionTree)
    node = tree.root
    io_new = IOContext(io, :level => 1)
    print(io_new, node)
end

function Base.show(io::IO, node::Node)
    indentation = ""
    level = get(io, :level, 1)
    for _ in 1:level
        indentation *= "--"
    end

    io_new = IOContext(io, :level => level + 1)

    println("$indentation Feature Index: $(node.feature_index)")
    println("$indentation Split Value: $(node.split_value)")
    println("$indentation-- Left")
    print(io_new, node.left)
    println("$indentation-- Right")
    print(io_new, node.right)
end

function Base.show(io::IO, leaf::Leaf)
    indentation = ""
    level = get(io, :level, 1)
    for _ in 1:level
        indentation *= "--"
    end
    println("$indentation Labels: $(leaf.values)")
end