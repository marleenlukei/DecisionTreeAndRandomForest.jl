"""
Represents a Leaf in the DecisionTree structure.

`values` stores the labels of the data points.
"""
mutable struct Leaf{L}
    values::Vector{L} # represents the prediction/s
end

"""
Represents a Node in the DecisionTree structure.

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
Represents a DecisionTree.

`max_depth` controls the maximum depth of the tree. If -1, the depth is not limited.

`min_samples_split` controls when a node in the decision tree should be split.

`root` contains the root Node of the DecisionTree.

`data` contains the datapoints of the DecisionTree.

`labels` contains the respective labels of the datapoints.
"""
mutable struct DecisionTree{T, L}
    """
    If the max_depth is -1, the DecisionTree is of unlimited depth.
    """
    max_depth::Int
    min_samples_split::Int
    split_criterion::Function
    root::Union{Node{T, L}, Leaf{L}, Missing}
    data::Matrix{T}
    labels::Vector{L}
    
    DecisionTree(max_depth::Int, min_samples_split::Int, split_criterion::Function, data::Matrix{T}, labels::Vector{L}) where {T, L} = 
        size(data, 1) == length(labels) ? new{T, L}(max_depth, min_samples_split, split_criterion, missing, data, labels) : throw(ArgumentError("The number of rows in data must match the number of elements in labels"))
end

DecisionTree(data, labels, split_criterion) = DecisionTree(-1, 1, split_criterion, data, labels)


"""
    build_tree(data, labels, max_depth, min_samples_split, depth)

Build the tree structure of the DecisionTree

If `depth` is unspecified, it is set to 0
"""
function build_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, split_criterion::Function, depth::Int=0, num_features::Int=-1) where {T, L}    
    # If max_depth is reached or if the data can not be split further, return a leaf
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth) 
        return Leaf{L}(labels)                                                       
    end

    # if all the labels are the same, return a leaf
    if allequal(labels) 
        return Leaf{L}(labels)
    end
    
    # Get the best split from the respective split_criterion
    feature_index, split_value = split_criterion(data, labels, num_features)
    # Random values for testing purposes
    # feature_index = rand((1:size(data, 2)))
    # split_value = data[rand((1:size(data, 1))), feature_index]
    if feature_index == 0
        return Leaf{L}(labels)
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
        return Leaf{L}(labels)
    end

    # Create the node with the computed attributes
    node = Node(data, labels)
    node.feature_index = feature_index
    node.split_value = split_value

    # Build the subtrees for the node
    node.left = build_tree(left_data, left_labels, max_depth, min_samples_split, split_criterion, depth + 1, num_features)
    node.right = build_tree(right_data, right_labels, max_depth, min_samples_split, split_criterion, depth + 1, num_features)

    # Return the node
    return node
end

"""
    fit(tree::DecisionTree)

Compute the tree structure.
"""
function fit(tree::DecisionTree, num_features::Int=-1)
    tree.root = build_tree(tree.data, tree.labels, tree.max_depth, tree.min_samples_split, tree.split_criterion, 0, num_features)
end

"""
    predict(tree::DecisionTree, data::Matrix{T})

Returns the prediction of the DecisionTree for a list of datapoints.

`data` contains the datapoints to predict.
"""
function predict(tree::DecisionTree, data::Matrix{T}) where {T}    
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
        if all(isa.(node.values, Number))
            push!(predictions, mean(node.values))
        else
            push!(predictions, mode(node.values))
        end
    end
    return predictions
end

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
    println("$indentation Data for index: $(node.data[:, node.feature_index])")
    println("$indentation Labels: $(node.labels)")
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