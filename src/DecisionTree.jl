"""
Represents a Leaf in the DecisionTree structure.

`values` stores the labels of the data points.
"""
mutable struct Leaf
    values::Vector

    Leaf(values::Vector) = new(values)
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
mutable struct Node    
    left::Union{Node, Leaf}
    right::Union{Node, Leaf}

    feature_index::Int
    split_value

    Node(left::Union{Node, Leaf}, right::Union{Node, Leaf}, feature_index::Int, split_value) = new(left, right, feature_index, split_value)
end

"""
Represents a DecisionTree.

`max_depth` controls the maximum depth of the tree. If -1, the depth is not limited.

`min_samples_split` controls when a node in the decision tree should be split.

`root` contains the root Node of the DecisionTree.

`data` contains the datapoints of the DecisionTree.

`labels` contains the respective labels of the datapoints.
"""
mutable struct DecisionTree
    """
    If the max_depth is -1, the DecisionTree is of unlimited depth.
    """
    max_depth::Int
    min_samples_split::Int
    num_features::Int
    split_criterion::Function

    root::Union{Node, Leaf, Missing}
    
    DecisionTree(max_depth::Int, min_samples_split::Int, num_features::Int, split_criterion::Function) = new(max_depth, min_samples_split, num_features, split_criterion, missing)
end

DecisionTree(split_criterion) = DecisionTree(-1, 1, -1, split_criterion)
DecisionTree(max_depth, min_samples_split, split_criterion) = DecisionTree(max_depth, min_samples_split, -1, split_criterion)


"""
    build_tree(data, labels, max_depth, min_samples_split, depth)

Build the tree structure of the DecisionTree

If `depth` is unspecified, it is set to 0
"""
function build_tree(data::Matrix, labels::Vector, max_depth::Int, min_samples_split::Int, num_features::Int, split_criterion::Function, depth::Int=0)
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
    # Random values for testing purposes
    # feature_index = rand((1:size(data, 2)))
    # split_value = data[rand((1:size(data, 1))), feature_index]
    if feature_index == 0
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
    fit(tree::DecisionTree)

Compute the tree structure.
"""
function fit(tree::DecisionTree, data::Matrix, labels::Vector)
    if size(data, 1) != length(labels)
        throw(ArgumentError("The number of rows in data must match the number of elements in labels -> $(size(data, 1)) != $(length(labels))"))
    end 
    tree.root = build_tree(data, labels, tree.max_depth, tree.min_samples_split, tree.num_features, tree.split_criterion, 0)
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