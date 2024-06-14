module DecisionTreeAndRandomForest



# Write your package code here.
include("ClassificationTree.jl")
include("GiniImpurity.jl")
include("InformationGain.jl")

export ClassificationTree, fit, predict, print_tree,split_gini,split_ig
end
