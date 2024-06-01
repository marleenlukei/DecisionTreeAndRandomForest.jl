mutable struct Leaf{L}
    values::Vector{L}
end

mutable struct Node{T, L}
    # I would like to have it typed like this, but then it can not be initialized with 'missing'
    # left::Union{Node{T, L}, Leaf{L}}
    # right::Union{Node{T, L}, Leaf{L}}
    left
    right

    feature_index::Int
    split_value::T

    data::Matrix{T}
    labels::Vector{L}

    Node(data::Matrix{T}, labels::Vector{L}) where {T, L} = new{T, L}(missing, missing, missing, missing, data, labels)
end

"""
max_depth describes the depth that is left from this point on.
If max_depth is -1 the depth is unlimited.
"""
function fit(node::Node, splitting_criterion::Function, max_depth::Int)
    # TODO
    # 1. evaluate the best split using the splitting_criterion function
    # 2. assign the returned feature and value to the attributes in the node
    # 3. create the left and right subtrees
    # 4. call fit on the subtrees with the respective data left until max_depth is reached or at a leaf
end

mutable struct ClassificationTree{T, L}
    """
    If the max_depth is -1, the DecisionTree is of unlimited depth.
    """
    max_depth::Int

    root
    # I would like to have it typed like this, but then it can not be initialized with 'missing'
    # root::Union{Node{T, L}, Leaf{L}}

    data::Matrix{T}
    labels::Vector{L}

    # TODO check if the shapes of data and labels match
    ClassificationTree(max_depth::Int, data::Matrix{T}, labels::Vector{L}) where {T, L} = new{T, L}(max_depth, missing, data, labels)
end

ClassificationTree(data, labels) = ClassificationTree(-1, data, labels)

function fit(tree::ClassificationTree, splitting_criterion::Function)
    tree.root = Node(tree.data, tree.labels)
    fit(tree.root, splitting_criterion, tree.max_depth - 1)
end

function predict(tree::ClassificationTree, data::Matrix{T}) where {T}
    # TODO
    # 1. for each sample, go through the tree and save the resulting labels
    # 2. return the labels
end