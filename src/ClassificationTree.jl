include("InformationGain.jl")
include("GiniImpurity.jl")

using StatsBase: mode

# Represents a Leaf in the ClassificationTree structure.
mutable struct Leaf{L}
    values::Vector{L} # represents the prediction/s
end

# Represents a Node in the ClassificationTree structure.
mutable struct Node{T, L}    
    left::Union{Node{T, L}, Leaf{L}, Missing}
    right::Union{Node{T, L}, Leaf{L}, Missing}

    feature_index::Int
    split_value::Union{T, Missing}

    data::Matrix{T}
    labels::Vector{L}

    Node(data::Matrix{T}, labels::Vector{L}) where {T, L} = new{T, L}(missing, missing, -1, missing, data, labels)
end

# Represents a ClassificationTree.
mutable struct ClassificationTree{T, L}
    max_depth::Int
    min_samples_split::Int
    split_criterion::Function
    root::Union{Node{T, L}, Leaf{L}, Missing}
    data::Matrix{T}
    labels::Vector{L}
    
    ClassificationTree(max_depth::Int, min_samples_split::Int, split_criterion::Function, data::Matrix{T}, labels::Vector{L}) where {T, L} = 
        size(data, 1) == length(labels) ? new{T, L}(max_depth, min_samples_split, split_criterion, missing, data, labels) : throw(ArgumentError("The number of rows in data must match the number of elements in labels"))
end

ClassificationTree(data, labels, split_criterion) = ClassificationTree(-1, 1, split_criterion, data, labels)

# Build the tree structure of the ClassificationTree
function build_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, split_criterion::Function, depth::Int=0) where {T, L}
    # If max_depth is reached or if the data can not be split further, return a leaf
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth) 
        return Leaf{L}(labels)
    end

    # if all the labels are the same, return a leaf
    if allequal(labels)
        return Leaf{L}(labels)
    end

    # Get the best split from the provided split criterion
    feature_index, split_value = split_criterion(data, labels)

    # Handle the case when no valid split is found
    if feature_index == -1
        return Leaf{L}(labels)
    end

    # Create the mask on the data
    if isa(split_value, Number)
        left_mask = data[:, feature_index] .< split_value
        right_mask = data[:, feature_index] .>= split_value
    else
        left_mask = data[:, feature_index] .!= split_value
        right_mask = data[:, feature_index] .== split_value
    end

    # Compute the data and labels for the child nodes
    left_data, right_data = data[left_mask, :], data[right_mask, :]
    left_labels, right_labels = labels[left_mask], labels[right_mask]

    # Create the node with the computed attributes
    node = Node(data, labels)
    node.feature_index = feature_index
    node.split_value = split_value

    # Build the subtrees for the node
    node.left = build_tree(left_data, left_labels, max_depth, min_samples_split, split_criterion, depth + 1)
    node.right = build_tree(right_data, right_labels, max_depth, min_samples_split, split_criterion, depth + 1)

    # Return the node
    return node
end

# Compute the tree structure
function fit(tree::ClassificationTree)
    tree.root = build_tree(tree.data, tree.labels, tree.max_depth, tree.min_samples_split, tree.split_criterion)
end

# Returns the prediction of the ClassificationTree for a list of datapoints.
function predict(tree::ClassificationTree, data::Matrix{T}) where {T}
    predictions = []

    for i in 1:size(data, 1)
        node = tree.root
        while !isa(node, Leaf)
            if isa(node.split_value, Number)
                if data[i, node.feature_index] < node.split_value
                    node = node.left
                else
                    node = node.right
                end
            else
                if data[i, node.feature_index] != node.split_value
                    node = node.left
                else
                    node = node.right
                end
            end
        end
        # Get the label that occurs the most and add it to predictions
        push!(predictions, mode(node.values))
    end
    return predictions
end

# Prints the tree structure. Mainly used for debugging purposes.
function print_tree(tree::ClassificationTree)
    node = tree.root
    level = 1
    print(node, level)
end

function print(node::Node, level::Int)
    indentation = ""
    for i in 1:level
        indentation *= "--"
    end

    println("$indentation Feature Index: $(node.feature_index)")
    println("$indentation Data for index: $(node.data[:, node.feature_index])")
    println("$indentation Labels: $(node.labels)")
    println("$indentation Split Value: $(node.split_value)")
    println("$indentation-- Left")
    print(node.left, level + 1)
    println("$indentation-- Right")
    print(node.right, level + 1)
end

function print(leaf::Leaf, level::Int)
    indentation = ""
    for i in 1:level
        indentation *= "--"
    end
    println("$indentation Labels: $(leaf.values)")
end
