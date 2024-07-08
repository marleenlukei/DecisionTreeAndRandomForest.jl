"""
    $(SIGNATURES)

Retrieves the implemented split criterions that can be used.

## Arguments
- `task::String`: A String that indicates for which task the splitting criterions should be retrieved. Can be `"classification"` or `"regression"`.
    Defaults to returning all available criterions

## Returns
- `Tuple{Function}`: A tuple containing the implemented functions.
  """
function get_split_criterions(task::String="all")
    criterions = Dict(
        "classification" => (split_gini, split_ig),
        "regression" => (split_variance,)
    )
    if task == "classification" || task == "regression"
        return criterions[task]
    end
    return (criterions["classification"]..., criterions["regression"]...)
end



"""
  $(SIGNATURES)

This function splits the labels into two subsets based on the provided feature and value. It handles both numerical and categorical features.

## Arguments
- `X::AbstractMatrix{T}`: A matrix of features, where each row is a data point and each column is a feature.
- `y::AbstractVector`: A vector of labels corresponding to the data points.
- `index::Int`: The index of the feature to split on.
- `value::T`: The value to split the feature on.

## Returns
- `Tuple{AbstractVector, AbstractVector}`: A tuple containing the left and right sets of labels.
"""
function split_node(X::AbstractMatrix{T}, y::AbstractVector, index::Int, value::T) where {T}
    x_index = X[:, index]
    # if feature is numerical
    if eltype(identity.(x_index)) <: Number
        mask = x_index .>= value
        # if feature is categorical
    else
        mask = x_index .== value
    end
    left = (y[.!mask])
    right = (y[mask])
    return left, right
end