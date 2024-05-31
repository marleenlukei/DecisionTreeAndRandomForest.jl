mutable struct Leaf{L}
    values::Vector{L}
end

mutable struct Node{T, L}
    left::Union{Node{T, L}, Leaf{L}}
    right::Union{Node{T, L}, Leaf{L}}

    feature_index::Int
    split_value::T
end

function fit(node::Node, splitting_criterion::Function, data::Matrix{T}, labels::Vector{L}) where {T, L}
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

    left::Union{Node{T, L}, Leaf{L}}
    right::Union{Node{T, L}, Leaf{L}}

    feature_index::Int
    split_value::T

    data::Matrix{T}
    labels::Vector{L}

    ClassificationTree(max_depth::Int, data::Matrix{T}, labels::Vector{L}) where {T, L} = new{T, L}(max_depth, Leaf{L}([]), Leaf{L}([]), -1, zero(T), data, labels)
end

ClassificationTree(data::Matrix{T}, labels::Vector{L}) where {T, L} = ClassificationTree(-1, data, labels)

function fit(tree::ClassificationTree, splitting_criterion::Function, data::Matrix{T}, labels::Vector{L}) where {T, L}
    # TODO
    # 1. evaluate the best split using the splitting_criterion function
    # 2. assign the returned feature and value to the attributes in the tree
    # 3. create the left and right subtrees
    # 4. call fit on the subtrees with the respective data left until max_depth is reached or at a leaf
end

function predict(tree::ClassificationTree, data::Matrix{T}) where {T}
    # TODO
    # 1. for each sample, go through the tree and save the resulting labels
    # 2. return the labels
end