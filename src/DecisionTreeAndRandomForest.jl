module DecisionTreeAndRandomForest



# Write your package code here.
include("ClassificationTree.jl")
include("GiniImpurity.jl")
include("InformationGain.jl")
include("VarianceReduction.jl")

export ClassificationTree, fit, predict, print_tree, find_best_split
end
