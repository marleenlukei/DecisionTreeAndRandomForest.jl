module DecisionTreeAndRandomForest



# Write your package code here.
include("ClassificationTree.jl")
include("GiniImpurity.jl")
include("InformationGain.jl")
include("RandomForest.jl")

export ClassificationTree, fit, predict, print_tree, find_best_split, weighted_gini, split_node, RandomForest
end
