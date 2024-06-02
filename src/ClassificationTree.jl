mutable struct Leaf{L}
    values::Vector{L} # represents the prediction/s
end

mutable struct Node{T, L}
    # I would like to have it typed like this, but then it can not be initialized with 'missing' 
    # left::Union{Node{T, L}, Leaf{L}}
    # right::Union{Node{T, L}, Leaf{L}}
    
    # - Maybe you add a Missing type or nothing(not sure for this) at the Union for example: 
    left::Union{Node{T, L}, Leaf{L}, Missing}
    right::Union{Node{T, L}, Leaf{L}, Missing}
    
    # left
    # right

    feature_index::Int # Index of the feature (or attribute) on which the data is split at this node
    split_value::T # value of the feauture/ attributes used to split the decision node of type T

    data::Matrix{T}
    labels::Vector{L}

    Node(data::Matrix{T}, labels::Vector{L}) where {T, L} = new{T, L}(missing, missing, -1, zero(T), data, labels)
end


"""
Another idea for the node structure. The leaf node is included in the sctructure

mutable struct Node{T, L}
    feature::Union{Int, Nothing}  # Index of the feature on which the data is split at this node or nothing for leaf node
    split_value::Union{T, Nothing}  # split value value for splitting (internal node) or nothing (leaf node)
    left::Union{Node{T, L}, Nothing}  # left child (or nothing for leaf node)
    right::Union{Node{T, L}, Nothing}  # right child (or nothing for leaf node)
    prediction_value::Union{Vector{L}, Nothing}  # prediction value (for leaf node) or nothing (for internal node)

    # Constructor for leaf nodes
    Node(prediction_value::L) where {T, L} = new{T, L}(nothing, nothing, nothing, nothing, prediction_value)

    # Constructor for internal nodes
    Node(feature::Int, split_value::T, left::Node{T, L}, right::Node{T, L}) where {T, L} = new{T, L}(feature, split_value, left, right, nothing)
end
"""

"""
max_depth describes the depth that is left from this point on.
If max_depth is -1 the depth is unlimited.
"""

mutable struct ClassificationTree{T, L}
    """
    If the max_depth is -1, the DecisionTree is of unlimited depth.
    """
    max_depth::Int
    min_samples_split::Int #  controls when a node in the decision tree should be split (I think addind this is also good practice to avoid overfitting)

    # root
    # I would like to have it typed like this, but then it can not be initialized with 'missing'
    # # - Maybe you add a Missing type or nothing(not sure for this) at the Union for example: root::Union{Node{T, L}, Leaf{L}, Missing}

    # root::Union{Node{T, L}, Leaf{L}}

    root::Union{Node{T, L}, Leaf{L}, Missing}

    data::Matrix{T}
    labels::Vector{L}
    
    # Here is checked if the shapes of data and labels match
    ClassificationTree(max_depth::Int, min_samples_split::Int, data::Matrix{T}, labels::Vector{L}) where {T, L} = 
        size(data, 1) == length(labels) ? new{T, L}(max_depth, min_samples_split, missing, data, labels) : throw(ArgumentError("The number of rows in data must match the number of elements in labels"))


end

ClassificationTree(data, labels) = ClassificationTree(-1, 1, data, labels)

function build_tree(data::Matrix{T}, labels::Vector{L}, max_depth::Int, min_samples_split::Int, depth::Int=0) where {T, L}
    
    if length(labels) < min_samples_split || (max_depth != -1 && depth >= max_depth) # checks if the node can be split further (1) and if the depth has exceded max_depth (2)
        return Leaf{L}(labels)                                                       # if max depth is -1 (unlimited length) -> depth is always greater that max_depth, but should be ignored
    end
    
    # Get feature index and split value from find best split - I think Marieen had already implemented this so we can just import it
    feature_index, split_value = 1, 3 # find_best_split(data, labels) # used some values just for testing

    # added this because I was getting an error - might delete later
    if feature_index == -1
        return Leaf{L}(labels)
    end

    # Split the data
    left_indices = data[:, feature_index] .<= split_value
    right_indices = data[:, feature_index] .> split_value

    # added this because I was getting an error - might delete
    if isempty(left_indices) || isempty(right_indices)
        return Leaf{L}(labels)
    end

    left_data, right_data = data[left_indices, :], data[right_indices, :]
    left_labels, right_labels = labels[left_indices], labels[right_indices]

    node = Node(data, labels)
    node.feature_index = feature_index
    node.split_value = split_value
    node.left = build_tree(left_data, left_labels, max_depth, min_samples_split, depth + 1)
    node.right = build_tree(right_data, right_labels, max_depth, min_samples_split, depth + 1)

    return node
end

# function fit(tree::ClassificationTree, find_best_split::Function)
#    tree.root = Node(tree.data, tree.labels)
#    build_tree(tree.root, tree.max_depth, tree.min_samples_split, 0)

function fit(tree::ClassificationTree)
    tree.root = build_tree(tree.data, tree.labels, tree.max_depth, tree.min_samples_split)
end


function predict(tree::ClassificationTree, data::Matrix{T}) where {T}
    # TODO
    # 1. for each sample, go through the tree and save the resulting labels
    # 2. return the labels
    
    predictions = Vector{L}[] # to store the predictions
    
    for i in 1:size(data, 1) # go through each sample
        node = tree.root
        while !isa(node, Leaf)
            if data[i, node.feature_index] <= node.split_value
                node = node.left
            else
                node = node.right
            end
        end
        push!(predictions, node.values)
    end
    return predictions
end