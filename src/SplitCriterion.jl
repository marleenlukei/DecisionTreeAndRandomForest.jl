module SplitCriterion

include("InformationGain.jl")
include("GiniImpurity.jl")


function find_best_split(X::AbstractMatrix{T}, y::Vector{L}, criterion::String) where {T, L}
    if criterion == "IG"
        return best_split(X, y)
    elseif criterion == "GI"
        return find_best_split_gini(X, y)
    else
        throw(ArgumentError("Invalid criterion. Use 'IG' for Information Gain or 'GI' for Gini Impurity."))
    end
end

export find_best_split

end
