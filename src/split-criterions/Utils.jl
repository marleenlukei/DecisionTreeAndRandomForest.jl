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