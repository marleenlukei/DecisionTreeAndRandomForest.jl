using StatsBase: mode

"""
Represents a Leaf in the ClassificationTree structure.

`values` stores the labels of the data points.
"""
mutable struct Leaf{L}
    values::Vector{L} # represents the prediction/s
end

"""
Represents a Node in the ClassificationTree structure.

`left` points to the left child.

`right` points to the right child.

`feature_index` stores the index of the selected feature.

`split_value` stores the value on that the data is split.

`data` contains the datapoints of the Node.

`labels` contains the respective labels of the datapoints.
"""
mutable struct Node{T, L}    
    left::Union{Node{T, L}, Leaf{L}, Missing}
    right::Union{Node{T, L}, Leaf{L}, Missing}

    feature_index::Int
    split_value::Union{T, Missing}

    data::Matrix{T}
    labels::Vector{L}

    Node(data::Matrix{T}, labels::Vector{L}) where {T, L} = new{T, L}(missing, missing, -1, missing, data, labels)
end

"""
Represents a ClassificationTree.

`max_depth` controls the maximum depth of the tree. If -1, the depth is not limited.

`min_samples_split` controls when a node in the decision tree should be split.

`root` contains the root Node of the ClassificationTree.

`data` contains the datapoints of the ClassificationTree.

`labels` contains the respective labels of the datapoints.
"""
mutable struct ClassificationTree{T, L}
    """
    If the max_depth is -1, the DecisionTree is of unlimited depth.
    """
    max_depth::Int
    min_samples_split::Int

    root::Union{Node{T, L}, Leaf{L}, Missing}

    data::Matrix{T}
    labels::Vector{L}
    
    ClassificationTree(max_depth::Int, min_samples_split::Int, data::Matrix{T}, labels::Vector{L}) where {T, L} = 
        size(data, 1) == length(labels) ? new{T, L}(max_depth, min_samples_split, missing, data, labels) : throw(ArgumentError("The number of rows in data must match the number of elements in labels"))


end

ClassificationTree(data, labels) = ClassificationTree(-1, 1, data, labels)


"""
    build_tree(data, labels, max_depth, min_samples_split, depth)

Build the tree structure of the ClassificationTree

If `depth` is unspecified, it is set to 0
"""
function build_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, depth::Int=0) where {T, L}
    
    # If max_depth is reached or if the data can not be split further, return a leaf
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth) 
        return Leaf{L}(labels)                                                       
    end
    
    # Get the best split from the respective split_criterion
    feature_index, split_value = find_best_split(data, labels)
    # Random values for testing purposes
    # feature_index = rand((1:size(data, 2)))
    # split_value = data[rand((1:size(data, 1))), feature_index]

    # Create the mask on the data
    if isa(split_value, Number)
        left_mask = data[:, feature_index] .< split_value
        right_mask = data[:, feature_index] .>= split_value
    else
        left_mask = data[:, feature_index] .!= split_value
        right_mask = data[:, feature_index] .== split_value
    end

    # If the data can not be split further, return a leaf
    if allequal(left_mask) || allequal(right_mask)
        return Leaf{L}(labels)
    end

    # Compute the data and labels for the child nodes
    left_data, right_data = data[left_mask, :], data[right_mask, :]
    left_labels, right_labels = labels[left_mask], labels[right_mask]

    # Create the node with the computed attributes
    node = Node(data, labels)
    node.feature_index = feature_index
    node.split_value = split_value

    # Build the subtrees for the node
    node.left = build_tree(left_data, left_labels, max_depth, min_samples_split, depth + 1)
    node.right = build_tree(right_data, right_labels, max_depth, min_samples_split, depth + 1)

    # Return the node
    return node
end

"""
    fit(tree::ClassificationTree)

Compute the tree structure.
"""
function fit(tree::ClassificationTree)
    tree.root = build_tree(tree.data, tree.labels, tree.max_depth, tree.min_samples_split)
end

"""
    predict(tree::ClassificationTree, data::Matrix{T})

Returns the prediction of the ClassificationTree for a list of datapoints.

`data` contains the datapoints to predict.
"""
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

"""
    print_tree(tree:ClassificationTree)

Prints the tree structure. Mainly used for debugging purposes.
"""
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