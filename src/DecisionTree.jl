"""
    $(SIGNATURES)

Represents a Leaf in the DecisionTree structure.

## Fields
$(TYPEDFIELDS)
"""
mutable struct Leaf{T<:AbstractVector}
    values::T

    Leaf(values::T) where {T<:AbstractVector} = new{T}(values)
end

"""
    $(SIGNATURES)

Represents a Node in the DecisionTree structure.

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

Represents a DecisionTree.

## Fields
$(TYPEDFIELDS)
"""
mutable struct DecisionTree
    "Controls the maximum depth of the tree. If -1, the DecisionTree is of unlimited depth."
    max_depth::Int
    "Controls the minimum number of samples required to split a node."
    min_samples_split::Int
    "Controls the number of features to consider for each split. If -1, all features are used."
    num_features::Int
    "Contains the split criterion function."
    split_criterion::Function
    "Contains the root node of the DecisionTree."
    root::Union{Node,Leaf,Missing}

    DecisionTree(max_depth::Int, min_samples_split::Int, num_features::Int, split_criterion::Function) = new(max_depth, min_samples_split, num_features, split_criterion, missing)
end

DecisionTree(split_criterion::Function) = DecisionTree(-1, 1, -1, split_criterion)
DecisionTree(max_depth::Int, min_samples_split::Int, split_criterion::Function) = DecisionTree(max_depth, min_samples_split, -1, split_criterion)


"""
    $(SIGNATURES)

This function recursively builds a DecisionTree by iteratively splitting the data based on the provided `split_criterion`. The process continues until either the maximum depth is reached, the number of samples in a node falls below `min_samples_split` or all labels in a node are the same.

## Arguments
- `data::AbstractMatrix`: The training data.
- `labels::AbstractVector`: The labels for the training data.
- `max_depth::Int`: The maximum depth of the tree.
- `min_samples_split::Int`: The minimum number of samples required to split a node.
- `split_criterion::Function`: The function used to determine the best split at each node.
- `depth::Int=0`: The current depth of the tree (used recursively).

## Returns
- `Union{Node, Leaf}`: The return value can be one of two types, depending on the state of the tree at each point of recursion.
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

This function builds the tree structure of the `DecisionTree` by calling the `build_tree` function. 

## Arguments
- `tree::DecisionTree`: The DecisionTree to fit.
- `data::AbstractMatrix`: The training data.
- `labels::AbstractVector`: The labels for the training data.

## Returns
- `Nothing`: This function modifies the `tree` in-place.
"""
function fit!(tree::DecisionTree, data::AbstractMatrix, labels::AbstractVector)
    if size(data, 1) != length(labels)
        throw(ArgumentError("The number of rows in data must match the number of elements in labels -> $(size(data, 1)) != $(length(labels))"))
    end
    if !(eltype(labels) <: Number)
        if tree.split_criterion in get_split_criterions("regression")
            throw(ArgumentError("The chosen split criterion does only work for classification task, please choose another one"))
        end
    end
    tree.root = build_tree(data, labels, tree.max_depth, tree.min_samples_split, tree.num_features, tree.split_criterion, 0)
end

"""
    $(SIGNATURES)

This function traverses the tree structure of the `DecisionTree` for each datapoint in `data`. It follows the decision rules based on the split criteria and feature values. If the leaf node contains numerical values, its treated as a regreesion problem and the prediction is the average of those values. If a leaf node contains numerical values, it is treated as a regression problem, and the prediction is the average of those values. If the leaf node contains categorical labels, it is treated as a classification problem, and the prediction is the most frequent label (mode) among the labels in the leaf node.

## Arguments
- `tree::DecisionTree`: The trained DecisionTree.
- `data::AbstractMatrix`: The datapoints to predict.

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

This function recursively prints the structure of the `DecisionTree`, providing information about each node and leaf. It's primarily used for debugging and visualizing the tree's structure.

## Arguments
- `io::IO`: The IO context to print the tree structure.
- `tree::DecisionTree`: The DecisionTree to print.

## Returns
- `Nothing`: This function prints the structure of the `DecisionTree`.
"""
function Base.show(io::IO, tree::DecisionTree)
    node = tree.root
    io_new = IOContext(io, :level => 1)
    print(io_new, node)
end

function Base.show(io::IO, node::Node)
    indentation = ""
    level = get(io, :level, 1)
    pairs = get(io, :pairs, Dict())
    for i in 2:level
        if i < level
            if pairs[i]
                indentation *= "│   "
            else
                indentation *= "    "
            end
        else
            if pairs[i]
                indentation *= "├── "
            else
                indentation *= "└── "
            end
        end
    end
    pairs_left = Dict(pairs..., level + 1 => true)
    pairs_right = Dict(pairs..., level + 1 => false)
    io_left = IOContext(io, :level => level + 1, :pairs => pairs_left)
    io_right = IOContext(io, :level => level + 1, :pairs => pairs_right)

    println("$(indentation)Feature: $(node.feature_index), Split Value: $(node.split_value)")
    print(io_left, node.left)
    print(io_right, node.right)
end

function Base.show(io::IO, leaf::Leaf)
    indentation = ""
    level = get(io, :level, 1)
    pairs = get(io, :pairs, Dict())
    for i in 2:level
        if i < level
            if pairs[i]
                indentation *= "│   "
            else
                indentation *= "    "
            end
        else
            if pairs[i]
                indentation *= "├── "
            else
                indentation *= "└── "
            end
        end
    end
    occs = countmap(leaf.values)
    str = "$(indentation)Labels: "
    for (key, value) in occs
        str *= "$key ($value/$(length(leaf.values))) "
    end
    println(str)
end